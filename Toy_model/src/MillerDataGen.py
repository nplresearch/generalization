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


# ========================= Compositional extension ========================= #

@dataclass
class CompositionalDataGeneratorConfig:
    """Configuration for compositional data generation with multiple feature slots"""
    num_slots: int = 2
    vocab_sizes: Tuple[int, ...] = (16, 16)  # size per slot
    sequence_len: int = 3
    batch_size: int = 32
    slot_metric_matrices: Tuple[torch.Tensor, ...] = ()  # one distance matrix per slot
    slot_weights: Tuple[float, ...] = ()  # optional weights per slot (unused for max-sim rule)
    seed: int = 43
    num_samples: int = 1000
    # Optional explicit train/test compositions as LongTensor [N, num_slots]
    train_compositions: Optional[torch.Tensor] = None
    test_compositions: Optional[torch.Tensor] = None
    # How to aggregate per-slot distances when creating labels: 'min' (default), 'max', 'L1', 'L2'
    distance_aggregation: str = "min"

class CompositionalDataGenerator:
    """
    Data generator where each stimulus is a composition of feature slots.
    Labels are computed by aggregating per-slot distances (min/max/L1/L2) per config.
    """
    def __init__(self, config: CompositionalDataGeneratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        torch.manual_seed(config.seed)

        assert config.num_slots >= 1, "num_slots must be >= 1"
        assert len(config.vocab_sizes) == config.num_slots, "vocab_sizes must match num_slots"
        assert len(config.slot_metric_matrices) == config.num_slots, "Need a metric matrix per slot"

        # Precompute slot offsets to build concatenated multi-hot vectors
        self.slot_offsets = np.cumsum([0] + list(config.vocab_sizes[:-1]))
        # For compatibility with downstream code that expects a `num_inputs` field, we expose
        # the total input dimensionality as num_inputs (basis features/primitives count)
        self.config.num_inputs = int(sum(config.vocab_sizes))  # type: ignore[attr-defined]

    # --------- Helpers ---------
    def _sample_composites(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of sequences of composite indices.
        Returns LongTensor of shape [batch_size, sequence_len, num_slots], each entry in [0, vocab_size_k-1].
        Ensures within-row uniqueness across context candidates for each slot when possible.
        """
        B = batch_size
        K = self.config.sequence_len
        S = self.config.num_slots
        seq = torch.zeros((B, K, S), dtype=torch.long)
        for k in range(S):
            V = int(self.config.vocab_sizes[k])
            # sample without replacement across the K candidates when feasible
            if V >= K:
                # For each batch item, draw a random permutation and take first K
                perms = torch.argsort(torch.rand(B, V), dim=1)[:, :K]
                seq[:, :, k] = perms
            else:
                # Fallback to with-replacement when vocab smaller than K
                seq[:, :, k] = torch.randint(high=V, size=(B, K))
        return seq

    def _composite_indices_to_inputs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert composite slot indices [B, seq_len, num_slots] to concatenated multi-hot inputs [B, seq_len, sum(V)].
        """
        batch_size, seq_len, num_slots = indices.shape
        total_dim = int(sum(self.config.vocab_sizes))
        xs = torch.zeros((batch_size, seq_len, total_dim), dtype=torch.float32)
        for k in range(self.config.num_slots):
            offset = int(self.slot_offsets[k])
            V = int(self.config.vocab_sizes[k])
            oh = one_hot(indices[:, :, k], V).to(torch.float32)
            xs[:, :, offset:offset+V] = oh
        return xs

    def _compute_aggregated_slot_distance(self, ctx: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        """
        Given ctx [B, K-1, num_slots], probe [B, 1, num_slots], compute per-candidate distance
        aggregated across slots according to self.config.distance_aggregation.
        Returns [B, K-1]. Lower is better.
        """
        per_slot_dists = []
        for k in range(self.config.num_slots):
            Dk = self.config.slot_metric_matrices[k]
            ctx_k = ctx[..., k].long()            # [B, K-1]
            probe_k = probe[..., k].squeeze(1).long()  # [B]
            # Select per-batch column corresponding to probe_k, then gather per-row ctx_k
            cols = Dk[:, probe_k]                  # [V_k, B]
            cols_bt = cols.permute(1, 0)          # [B, V_k]
            dk = cols_bt.gather(1, ctx_k)         # [B, K-1]
            dk = dk.unsqueeze(-1)                  # [B, K-1, 1]
            per_slot_dists.append(dk)
        d_stack = torch.cat(per_slot_dists, dim=2)  # [B, K-1, num_slots]

        agg = (self.config.distance_aggregation or "min").lower()
        if agg == "min":
            d_out, _ = torch.min(d_stack, dim=2)
        elif agg == "max":
            d_out, _ = torch.max(d_stack, dim=2)
        elif agg == "l1":
            d_out = torch.sum(d_stack, dim=2)
        elif agg == "l2":
            d_out = torch.sqrt(torch.sum(d_stack**2, dim=2) + 1e-12)
        else:
            # Fallback to min if unknown
            d_out, _ = torch.min(d_stack, dim=2)
        return d_out  # [B, K-1]

    # --------- Public API matching the original generator ---------
    def indices_to_inputs(self, indices, ntot=None):
        return self._composite_indices_to_inputs(indices)

    def generate_similarity_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = self.config.batch_size
        inputs = self._sample_composites(batch_size)  # [B, K, S]
        xs = inputs[:, :self.config.sequence_len-1, :]  # [B, K-1, S]
        probe = inputs[:, [self.config.sequence_len-1], :]  # [B, 1, S]
        d_agg = self._compute_aggregated_slot_distance(xs, probe)  # [B, K-1]
        labels = torch.argmin(d_agg, dim=1)
        return inputs, labels

    def generate_identification_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = self.config.batch_size
        inputs = self._sample_composites(batch_size)  # [B, K, S]
        xs = inputs[:, :self.config.sequence_len-1, :]
        # choose one context idx as the probe to ensure an exact match exists
        pick = torch.randint(low=0, high=self.config.sequence_len-1, size=(batch_size,))  # [B]
        probe = xs[torch.arange(batch_size), pick].unsqueeze(1)  # [B, 1, S]
        inputs[:, [self.config.sequence_len-1], :] = probe
        labels = pick  # [B]
        return inputs, labels
