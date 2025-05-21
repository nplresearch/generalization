import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from MillerDataGen import MillerDataGenerator

@dataclass
class MillerTestConfig:
    """Configuration for Miller's Law experiments"""
    input_dim: int = 32            # Dimension of input features
    feature_dim: int = 32          # Dimension of hidden featuress
    sequence_len: int = 3          # Length of sequence for similarity test
    num_samples: int = 1000        # Number of test samples
    epsilon: float = 0.5           # Similarity threshold
    batch_size: int = 32
    seed: int = 45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init_scale: float = 0.1        # Weight initialization scale 

class MillerTestEvaluator:
    """Evaluates model performance on Miller's Law tests"""
    def __init__(self, config: MillerTestConfig = MillerTestConfig()):
        self.config = config
    
    def evaluate_model(self, model: nn.Module, data_generator: MillerDataGenerator) -> Tuple[float, float]:
        """
        Evaluate model on both similarity and identification tests
        Returns:
            g_score: Generalization score (similarity test performance)
            i_score: Identification score (identification test performance)
        """
        model.eval()
        
        with torch.no_grad():
            # Similarity test
            correct_sim = 0
            total_sim = 0
            
            for _ in range(self.config.num_samples // self.config.batch_size):
                inputs, labels = data_generator.generate_similarity_test_batch()
                inputs = data_generator.indices_to_inputs(inputs).to(self.config.device)
                embs, __ = model(inputs)
                outputs = nn.functional.relu(torch.bmm(embs[:,:-1,:],embs[:,[-1],:].permute((0,2,1)) )) + 10**-10
                outputs = outputs.squeeze(2)
                # Max decision
                #pred = (torch.argmax(outputs,1)).float()
                #correct_sim += ((pred.cpu() )  == labels).sum().item()
                # Probability
                pred = outputs.float().cpu()
                pred = pred/torch.sum(pred,1,keepdims = True)
                correct_sim += (pred.gather(1,labels.unsqueeze(1))).sum().item()
                total_sim += labels.size(0)
            
            # Identification test
            correct_id = 0
            total_id = 0
            
            for _ in range(self.config.num_samples // self.config.batch_size):
                inputs, labels = data_generator.generate_identification_test_batch()
                inputs = data_generator.indices_to_inputs(inputs).to(self.config.device)
                embs, __= model(inputs)
                outputs = nn.functional.relu(torch.bmm(embs[:,:-1,:],embs[:,[-1],:].permute((0,2,1)) )) + 10**-10
                outputs = outputs.squeeze(2)
                # Max decision
                #pred = (torch.argmax(outputs,1)).float()
                #correct_id += ((pred.cpu()) == labels).sum().item()
                # Probability
                pred = outputs.float().cpu()
                pred = pred/torch.sum(pred,1,keepdims = True)

                correct_id += (pred.gather(1,labels.unsqueeze(1))).sum().item()
                total_id += labels.size(0)
        
        g_score = correct_sim / total_sim
        i_score = correct_id / total_id
        
        return g_score, i_score

class TransformerModel(nn.Module):
    """Simple transformer model for Miller's law testing"""
    def __init__(self, config: MillerTestConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed = nn.Linear(config.input_dim, config.feature_dim, bias = False)
        self.relu = nn.ReLU()    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)

        x = self.embed(x)
        #output = self.relu(torch.bmm(x[:,:-1,:],x[:,[-1],:].permute((0,2,1)) )) + 10**-10
        recon = self.relu(torch.matmul(x,self.embed.weight))

        return x, recon

def create_model(config: MillerTestConfig) -> TransformerModel:
    return TransformerModel(config)