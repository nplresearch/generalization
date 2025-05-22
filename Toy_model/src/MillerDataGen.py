import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict

from torch.nn.functional import one_hot

@dataclass
class DataGeneratorConfig:
    """Configuration for data generation"""
    num_inputs: int = 32          # Number of stimuli
    sequence_len: int = 3         # Length of sequence for similarity test
    batch_size: int = 32
    metric_matrix: torch.tensor = 1.0 - torch.eye(num_inputs) # Each stimulus is equidistant from the others
    seed: int = 43
    num_samples: int = 1000        # Total number of samples for training

class MillerDataGenerator:
    """Data generator for Miller's Law experiments"""
    def __init__(self, config: DataGeneratorConfig):
        # Store configuration and set random seeds for reproducibility
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        torch.manual_seed(config.seed)
        
    def _generate_base_features(self, batch_size: int) -> torch.Tensor:
        """Generate random indices for base features (with replacement)"""
        return torch.randint(high = self.config.num_inputs, size =  (batch_size,))
    
    def _generate_base_features_no_replace(self, batch_size: int) -> torch.Tensor:
        """Generate random indices for base features (without replacement)"""
        x = torch.rand(batch_size, self.config.num_inputs)
        indices = torch.argsort(torch.rand(*x.shape), dim=-1)  # Shuffle indices for each batch
        return indices[:, :self.config.sequence_len]  # Take first sequence_len indices

    def indices_to_inputs(self, indices, ntot = None):
        """Convert indices to one-hot encoded vectors"""
        if ntot is None:
            ntot = self.config.num_inputs
        return one_hot(indices, ntot).to(torch.float32)

    def generate_similarity_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of sequences (x1, x2, ..., xn, probe) for the similarity task.
        Returns:
            inputs: Tensor of shape (batch_size, sequence_len)
            labels: Tensor of shape (batch_size,) indicating closest element to probe
        """
        batch_size = self.config.batch_size
        
        # Generate unique input indices for each sample in the batch
        inputs = self._generate_base_features_no_replace(batch_size)

        xs = inputs[:, :self.config.sequence_len-1]  # First sequence_len-1 elements
        probe = inputs[:, [self.config.sequence_len-1]]  # Last element as probe

        # Compute distances between xs and probe using metric matrix
        d = self.config.metric_matrix[xs, probe]

        # Label is the index of the closest element in xs to the probe
        labels = torch.argmin(d, 1)

        return inputs, labels
    
    def generate_identification_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch for the identification task.
        Returns:
            inputs: Tensor of shape (batch_size, sequence_len)
            labels: Tensor of shape (batch_size,) indicating which element is the probe
        """
        batch_size = self.config.batch_size
        
        # Generate unique input indices for each sample in the batch
        inputs = self._generate_base_features_no_replace(batch_size)

        xs = inputs[:, :self.config.sequence_len-1]  # First sequence_len-1 elements

        # Shuffle xs and pick the last as probe
        indices = torch.argsort(torch.rand(*xs.shape), dim=-1)
        probe = xs[torch.arange(xs.shape[0]).unsqueeze(-1), indices][:, [-1]]
        inputs[:, [-1]] = probe  # Set probe as last element in inputs

        # Compute distances between xs and probe using metric matrix
        d = self.config.metric_matrix[xs, probe]
        # Label is the index of the closest element in xs to the probe
        labels = torch.argmin(d, 1)
        return inputs, labels
