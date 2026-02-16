# Toy Model API Reference

The `Toy_model/src/` package provides three modules for running controlled G-I tradeoff experiments with synthetic stimuli.

## MillerDataGen — Data Generators

### `DataGeneratorConfig`

Dataclass configuring the standard (flat) data generator.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_inputs` | `int` | 32 | Number of distinct stimuli |
| `sequence_len` | `int` | 3 | Context size (K-1 references + 1 probe) |
| `batch_size` | `int` | 32 | Batch size |
| `metric_matrix` | `torch.Tensor` | `1 - I` | Pairwise distance matrix (N x N). Defaults to equidistant stimuli |
| `seed` | `int` | 43 | Random seed |
| `num_samples` | `int` | 1000 | Total samples per epoch |

### `MillerDataGenerator(config)`

Generates batches for similarity and identification tasks on a flat stimulus space.

**Methods:**

- **`generate_similarity_test_batch()`** -> `(inputs, labels)`
  - `inputs`: `LongTensor [B, K]` — indices of K stimuli (last is probe)
  - `labels`: `LongTensor [B]` — index (within context) of the stimulus closest to the probe under the metric

- **`generate_identification_test_batch()`** -> `(inputs, labels)`
  - Same shape as above; the probe is guaranteed to be an exact copy of one context element
  - `labels` indicates which context element was duplicated

- **`indices_to_inputs(indices, ntot=None)`** -> `FloatTensor`
  - Converts integer indices to one-hot vectors of dimension `ntot` (defaults to `num_inputs`)

### `CompositionalDataGeneratorConfig`

Dataclass for multi-slot (compositional) stimuli.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_slots` | `int` | 2 | Number of feature slots |
| `vocab_sizes` | `Tuple[int, ...]` | (16, 16) | Vocabulary size per slot |
| `sequence_len` | `int` | 3 | Context size |
| `batch_size` | `int` | 32 | Batch size |
| `slot_metric_matrices` | `Tuple[Tensor, ...]` | — | One distance matrix per slot |
| `distance_aggregation` | `str` | `"min"` | How to combine per-slot distances: `"min"`, `"max"`, `"L1"`, `"L2"` |
| `seed` | `int` | 43 | Random seed |
| `num_samples` | `int` | 1000 | Total samples per epoch |

### `CompositionalDataGenerator(config)`

Same public API as `MillerDataGenerator`, but each stimulus is a tuple of per-slot features. Inputs are returned as concatenated multi-hot vectors of dimension `sum(vocab_sizes)`.

The `distance_aggregation` parameter controls how per-slot distances are combined to determine the similarity label:
- `"min"`: stimulus is close if it shares *any* slot value (most permissive)
- `"max"`: stimulus is close only if *all* slots are close (most restrictive)
- `"L1"` / `"L2"`: sum or Euclidean norm of per-slot distances

---

## MillerTest — Models and Evaluation

### `MillerTestConfig`

Dataclass for model configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dim` | `int` | 32 | Input vector dimension |
| `feature_dim` | `int` | 32 | Embedding dimension |
| `sequence_len` | `int` | 3 | Sequence length |
| `num_samples` | `int` | 1000 | Test samples |
| `batch_size` | `int` | 32 | Batch size |
| `activation` | `nn.Module` | `nn.ReLU()` | Activation function |
| `init_scale` | `float` | 0.1 | Weight initialization scale |

### `TransformerModel(config)`

Single linear projection (`input_dim` -> `feature_dim`, no bias). Similarity between stimuli is the ReLU of their inner product in embedding space.

```python
model = TransformerModel(MillerTestConfig(input_dim=32, feature_dim=16))
embeddings, reconstruction = model(x)  # x: [B, K, input_dim]
```

- `embeddings`: `[B, K, feature_dim]`
- `reconstruction`: `[B, K, input_dim]` — `activation(embeddings @ W)` where `W` is the embedding weight

### `CompositionalModel(config: CompositionalModelConfig)`

Per-slot linear projections. Similarity decisions use **max-over-slots** rule: two stimuli are similar if they share at least one slot with high inner-product overlap.

```python
from MillerTest import CompositionalModel, CompositionalModelConfig

config = CompositionalModelConfig(
    input_dim=32, feature_dim=16, sequence_len=3,
    vocab_sizes=(16, 16)
)
model = CompositionalModel(config)
embeddings, reconstruction = model(x)  # x: [B, K, sum(vocab_sizes)]
slot_embs = model.get_slot_embeddings(x)  # [B, K, num_slots, feature_dim]
```

### `MillerTestEvaluator(config)`

Evaluates a model on both G-score (similarity) and I-score (identification) tasks.

```python
evaluator = MillerTestEvaluator(config)
g_score, i_score, confusion_matrix = evaluator.evaluate_model(model, data_gen)
```

- `g_score`: fraction of similarity trials where the model correctly identifies the closest context element
- `i_score`: fraction of identification trials where the model correctly identifies the duplicated element
- `confusion_matrix`: `[N, N]` tensor tracking per-stimulus similarity predictions (non-compositional only)

---

## MillerTrainer — Training Loop

### `TrainerConfig`

Dataclass extending the loss configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `losses` | `list` | `["reconstruction", "similarity"]` | Active loss terms |
| `num_epochs` | `int` | 100 | Training epochs |
| `learning_rate` | `float` | 1e-3 | Adam learning rate |
| `weight_decay` | `float` | 1e-5 | L2 regularization |
| `patience` | `int` | 10 | LR scheduler patience |
| `min_lr` | `float` | 1e-6 | Minimum learning rate |
| `lambda_sim` | `float` | — | Weight for similarity loss (set in notebooks) |
| `lambda_rec` | `float` | — | Weight for reconstruction loss (set in notebooks) |

### `MillerTrainer(model, data_generator, config)`

Trains a model using two loss terms:

1. **Similarity loss** (NLL): encourages the model to rank the metrically closest context element highest
2. **Reconstruction loss** (MSE): prevents embedding collapse by reconstructing the one-hot input via the transpose embedding

```python
trainer = MillerTrainer(model, data_gen, TrainerConfig(num_epochs=200))

# Standard training
history = trainer.train(verbose=True, printevery=10)

# Training with embedding snapshots
history, embeddings = trainer.train(compute_embeddings=True)
```

**`train()` returns** a dict with keys:
- `similarity_scores` — G-score per epoch
- `identification_scores` — I-score per epoch
- `similarity_losses` — similarity loss per epoch
- `reconstruction_losses` — reconstruction loss per epoch
- `confusion_matrix_sim` — confusion matrices per epoch

The LR scheduler uses `ReduceLROnPlateau` maximising `G + I`.
