import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from MillerDataGen import MillerDataGenerator

@dataclass
class MillerTestConfig:
    """Configuration for Miller's Law experiments"""
    input_dim: int = 32            # Dimension of input vector
    feature_dim: int = 32          # Dimension of hidden features
    sequence_len: int = 3          # Length of sequence for similarity test
    num_samples: int = 1000        # Number of test samples
    epsilon: float = 0.5           # Similarity threshold
    batch_size: int = 32           # Batch size for evaluation
    seed: int = 45                 # Random seed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device selection
    init_scale: float = 0.1        # Weight initialization scale 
    activation: int = nn.ReLU()    # Activation function of the model

class MillerTestEvaluator:
    """Evaluates model performance on Miller's Law tests"""
    def __init__(self, config: MillerTestConfig = MillerTestConfig()):
        self.config = config
    
    def evaluate_model(self, model: nn.Module, data_generator: MillerDataGenerator) -> Tuple[float, float]:
        """
        Evaluate model on both similarity and identification tests.
        Returns:
            g_score: Generalization score (similarity test performance)
            i_score: Identification score (identification test performance)
        """
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Similarity test
            correct_sim = 0
            total_sim = 0
            confusion_matrix_sim = torch.zeros(self.config.num_inputs, self.config.num_inputs)
            number_appear_sim = torch.zeros(self.config.num_inputs, self.config.num_inputs)

            # Loop over batches for similarity test
            for _ in range(self.config.num_samples // self.config.batch_size):
                inputs, labels = data_generator.generate_similarity_test_batch()
                inputs_ids = inputs
                inputs = data_generator.indices_to_inputs(inputs).to(self.config.device)
                embs, __ = model(inputs)
                inputs_ids = inputs_ids.cpu()
                labels = labels.cpu()
                # Compute similarity outputs
                if isinstance(model.activation, torch.nn.ReLU):
                    outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1)))) + 10**-10
                else:
                    outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1))))
                outputs = outputs.squeeze(2)
            
                # Normalize predictions to probabilities
                pred = outputs.float().cpu()
                pred = pred / torch.sum(pred, 1, keepdims=True)

                # Update confusion matrices
                for b in range(self.config.batch_size):
                    confusion_matrix_sim[inputs_ids[b, labels[b]], inputs_ids[b, :-1]] += pred[b, :]
                    number_appear_sim[inputs_ids[b, labels[b]], inputs_ids[b, :-1]] += 1

                # Count correct predictions
                correct_sim += (pred.gather(1, labels.unsqueeze(1))).sum().item()
                total_sim += labels.size(0)
            
            # Normalize confusion matrix
            confusion_matrix_sim[number_appear_sim > 0] /= number_appear_sim[number_appear_sim > 0]

            # Identification test
            correct_id = 0
            total_id = 0
            
            # Loop over batches for identification test
            for _ in range(self.config.num_samples // self.config.batch_size):
                inputs, labels = data_generator.generate_identification_test_batch()
                inputs = data_generator.indices_to_inputs(inputs).to(self.config.device)
                embs, __ = model(inputs)

                # Compute identification outputs
                if isinstance(model.activation, torch.nn.ReLU):
                    outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1)))) + 10**-10
                else:
                    outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1))))
                outputs = outputs.squeeze(2)
               
                # Normalize predictions to probabilities
                pred = outputs.float().cpu()
                pred = pred / torch.sum(pred, 1, keepdims=True)

                # Count correct predictions
                correct_id += (pred.gather(1, labels.unsqueeze(1))).sum().item()
                total_id += labels.size(0)
        
        # Compute final scores
        g_score = correct_sim / total_sim
        i_score = correct_id / total_id
        confusion_matrix_sim = confusion_matrix_sim.cpu()
        return g_score, i_score, confusion_matrix_sim

class TransformerModel(nn.Module):
    """Simple transformer model for Miller's law testing"""
    def __init__(self, config: MillerTestConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer: projects input to feature dimension
        self.embed = nn.Linear(config.input_dim, config.feature_dim, bias=False)
        self.activation = config.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)
        x = self.embed(x)  # Project input
        # Compute reconstruction using activation and embedding weights
        recon = self.activation(torch.matmul(x, self.embed.weight))
        return x, recon

def create_model(config: MillerTestConfig) -> TransformerModel:
    """Factory function to create a TransformerModel"""
    return TransformerModel(config)