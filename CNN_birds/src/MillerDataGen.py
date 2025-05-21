import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict

from torch.nn.functional import one_hot

class TaskType(Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    MIXED = "mixed"

@dataclass
class DataGeneratorConfig:
    """Configuration for data generation"""
    num_inputs: int = 32          # Number of stimuli
    sequence_len: int = 3          # Length of sequence for similarity test
    batch_size: int = 32
    noise_level: float = 0.1       # Amount of noise to add
    metric_matrix: torch.tensor = 1.0 - torch.eye(num_inputs) # Each stimulus is equidistant from the others
    seed: int = 43
    num_samples: int = 1000        # Total number of samples for training


class MillerDataGenerator:
    """Enhanced data generator for Miller's Law experiments"""
    def __init__(self, config: DataGeneratorConfig):
        # Initialize configuration and random number generator
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        torch.manual_seed(config.seed)
        
    def _generate_base_features(self, batch_size: int) -> torch.Tensor:
        """Generate base feature vectors"""
        return torch.randint(high = self.config.num_inputs, size =  (batch_size,))
    
    def _generate_base_features_no_replace(self, batch_size: int) -> torch.Tensor:
        """Generate base feature vectors"""
        x = torch.rand(batch_size,self.config.num_inputs)
        indices = torch.argsort(torch.rand(*x.shape), dim=-1)

        return indices[:,:self.config.sequence_len]

    def _add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add controlled noise to tensor"""
        noise = torch.randn_like(tensor) * self.config.noise_level
        return tensor + noise
    
    def indices_to_inputs(self, indices, ntot = None):
        """Transform labels into one-hot vectors"""
        if ntot is None:
            ntot = self.config.num_inputs
        return one_hot(indices,ntot).to(torch.float32)

    def generate_similarity_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of triplets (x1, x2, probe) for similarity testing"""
        batch_size = self.config.batch_size
        
        # Generate indices
        inputs = self._generate_base_features_no_replace(batch_size)

        xs = inputs[:,:self.config.sequence_len-1]
        probe = inputs[:,[self.config.sequence_len-1]]

        # Pick closest elements to the probe
        d = self.config.metric_matrix[xs,probe]

        labels = torch.argmin(d,1)

        return inputs, labels
    
    def generate_identification_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch for identification testing"""
        batch_size = self.config.batch_size
        
        # Generate indices
        inputs = self._generate_base_features_no_replace(batch_size)

        xs = inputs[:,:self.config.sequence_len-1]
        #probe = inputs[:,[self.config.sequence_len-1]]
        # Pick random x as a probe
        indices = torch.argsort(torch.rand(*xs.shape), dim=-1)

        probe = xs[torch.arange(xs.shape[0]).unsqueeze(-1), indices][:,[-1]]
        inputs[:,[-1]] = probe 

        d = self.config.metric_matrix[xs,probe]
        labels = torch.argmin(d,1)
        return inputs, labels

    def generate_mixed_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate both similarity and identification batches"""
        sim_inputs, sim_labels = self.generate_similarity_test_batch()
        id_inputs, id_labels = self.generate_identification_test_batch()
        return sim_inputs, sim_labels, id_inputs, id_labels