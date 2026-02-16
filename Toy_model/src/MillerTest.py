"""
Model architectures and evaluation for the G-I tradeoff experiments.

Provides two model classes and an evaluator:

Models:
  - TransformerModel: A single linear projection (input_dim -> feature_dim)
    with no bias.  Similarity between stimuli is computed as the ReLU of
    their inner product in the embedding space.  This is the "non-slot"
    baseline model.
  - CompositionalModel: Independent per-slot linear projections whose
    embeddings are summed for the combined representation.  Similarity
    decisions use a *max-over-slots* rule: two stimuli are similar if they
    share at least one slot with high inner-product overlap.

Evaluator:
  - MillerTestEvaluator: Runs both the similarity (G-score) and
    identification (I-score) tasks over batches produced by a data generator.
    Also tracks a confusion matrix for the similarity task (non-compositional
    models only).

Both models expose the same forward signature:
    forward(x) -> (embeddings, reconstruction)
so they can be used interchangeably by the trainer and evaluator.
"""

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
            is_comp = getattr(model, 'is_compositional', False)
            if not is_comp:
                confusion_matrix_sim = torch.zeros(self.config.num_inputs, self.config.num_inputs)
                number_appear_sim = torch.zeros(self.config.num_inputs, self.config.num_inputs)
            else:
                # Placeholder small matrices when in compositional mode (not used downstream)
                confusion_matrix_sim = torch.zeros(1, 1)
                number_appear_sim = torch.zeros(1, 1)

            # Loop over batches for similarity test
            for _ in range(self.config.num_samples // self.config.batch_size):
                inputs, labels = data_generator.generate_similarity_test_batch()
                inputs_ids = inputs
                inputs = data_generator.indices_to_inputs(inputs).to(self.config.device)
                embs, __ = model(inputs)
                inputs_ids = inputs_ids.cpu()
                labels = labels.cpu()
                # Compute similarity outputs
                if getattr(model, 'is_compositional', False):
                    slot_embs = model.get_slot_embeddings(inputs)
                    # slot_embs: [B, K, S, F]
                    # per-slot logits: [B, K-1] for each slot, then max over slots
                    per_slot_logits = []
                    for s in range(slot_embs.shape[2]):
                        ctx = slot_embs[:, :-1, s, :]
                        probe = slot_embs[:, [-1], s, :]
                        logits_s = torch.bmm(ctx, probe.permute(0, 2, 1))  # [B, K-1, 1]
                        per_slot_logits.append(logits_s)
                    outputs = torch.max(torch.cat(per_slot_logits, dim=2), dim=2).values  # [B, K-1]
                    if isinstance(model.activation, torch.nn.ReLU):
                        outputs = model.activation(outputs) + 10**-10
                    else:
                        outputs = model.activation(outputs)
                else:
                    if isinstance(model.activation, torch.nn.ReLU):
                        outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1)))) + 10**-10
                    else:
                        outputs = model.activation(torch.bmm(embs[:, :-1, :], embs[:, [-1], :].permute((0, 2, 1))))
                    outputs = outputs.squeeze(2)
            
                # Normalize predictions to probabilities
                pred = outputs.float().cpu()
                pred = pred / torch.sum(pred, 1, keepdims=True)

                # Update confusion matrices
                if not is_comp:
                    for b in range(self.config.batch_size):
                        confusion_matrix_sim[inputs_ids[b, labels[b]], inputs_ids[b, :-1]] += pred[b, :]
                        number_appear_sim[inputs_ids[b, labels[b]], inputs_ids[b, :-1]] += 1

                # Count correct predictions
                correct_sim += (pred.gather(1, labels.unsqueeze(1))).sum().item()
                total_sim += labels.size(0)
            
            # Normalize confusion matrix
            if not is_comp:
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
                if getattr(model, 'is_compositional', False):
                    slot_embs = model.get_slot_embeddings(inputs)
                    per_slot_logits = []
                    for s in range(slot_embs.shape[2]):
                        ctx = slot_embs[:, :-1, s, :]
                        probe = slot_embs[:, [-1], s, :]
                        logits_s = torch.bmm(ctx, probe.permute(0, 2, 1))  # [B, K-1, 1]
                        per_slot_logits.append(logits_s)
                    outputs = torch.max(torch.cat(per_slot_logits, dim=2), dim=2).values  # [B, K-1]
                    if isinstance(model.activation, torch.nn.ReLU):
                        outputs = model.activation(outputs) + 10**-10
                    else:
                        outputs = model.activation(outputs)
                else:
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
        self.is_compositional: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)
        x = self.embed(x)  # Project input
        # Compute reconstruction using activation and embedding weights
        recon = self.activation(torch.matmul(x, self.embed.weight))
        return x, recon

    def get_slot_embeddings(self, x: torch.Tensor):
        """Non-compositional model has no slot embeddings."""
        return None

def create_model(config: MillerTestConfig) -> TransformerModel:
    """Factory function to create a TransformerModel"""
    return TransformerModel(config)


# ========================= Compositional extension ========================= #

@dataclass
class CompositionalModelConfig:
    input_dim: int  # sum of vocab sizes
    feature_dim: int
    sequence_len: int
    vocab_sizes: Tuple[int, ...]
    activation: nn.Module = nn.ReLU()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CompositionalModel(nn.Module):
    """
    Slot-wise linear embeddings. Input is concatenated multi-hot over slots.
    Similarity for decisions is computed as the max over per-slot similarities between candidate and probe.
    """
    def __init__(self, config: CompositionalModelConfig):
        super().__init__()
        self.config = config
        self.activation = config.activation
        self.vocab_sizes = list(config.vocab_sizes)
        self.num_slots = len(self.vocab_sizes)
        self.feature_dim = config.feature_dim
        self.is_compositional: bool = True

        # Per-slot linear projections (no bias)
        self.slot_embedders = nn.ModuleList([
            nn.Linear(v_size, config.feature_dim, bias=False) for v_size in self.vocab_sizes
        ])

        # Precompute slot offsets to slice inputs
        self.register_buffer(
            'slot_offsets',
            torch.tensor([0] + list(np.cumsum(self.vocab_sizes)[:-1]), dtype=torch.long),
            persistent=False
        )

    def _split_slots(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, K, sum(V)] -> list of [B, K, V_k]
        parts: List[torch.Tensor] = []
        start = 0
        for v in self.vocab_sizes:
            parts.append(x[:, :, start:start+v])
            start += v
        return parts

    def get_slot_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [B, K, S, F]
        parts = self._split_slots(x)
        slot_embs = []
        for k, part in enumerate(parts):
            # [B*K, V_k] @ [V_k, F] -> [B*K, F]
            B, K, V = part.shape
            part2d = part.reshape(B*K, V)
            emb2d = part2d @ self.slot_embedders[k].weight.t()
            emb = emb2d.reshape(B, K, self.feature_dim)
            slot_embs.append(emb.unsqueeze(2))
        return torch.cat(slot_embs, dim=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute per-slot embeddings and sum for combined embedding
        slot_embs = self.get_slot_embeddings(x)  # [B, K, S, F]
        embs = slot_embs.sum(dim=2)  # [B, K, F]
        # Reconstruction per slot, then concat
        recons = []
        parts = self._split_slots(x)
        for k in range(self.num_slots):
            emb_k = slot_embs[:, :, k, :]  # [B, K, F]
            # [B*K, F] @ [F, V_k] -> [B*K, V_k]
            B, K, F = emb_k.shape
            recon2d = emb_k.reshape(B*K, F) @ self.slot_embedders[k].weight
            recon_k = recon2d.reshape(B, K, self.vocab_sizes[k])
            recon_k = self.activation(recon_k)
            recons.append(recon_k)
        recon = torch.cat(recons, dim=2)  # [B, K, sum(V)]
        return embs, recon


def create_compositional_model(config: CompositionalModelConfig) -> CompositionalModel:
    return CompositionalModel(config)