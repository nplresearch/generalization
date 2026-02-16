# CNN Birds Experiment Guide

## Overview

The CNN_birds experiment trains ResNet-50 models on the CUB-200-2011 bird species dataset with a dual loss that controls the balance between species identification and phylogenetic generalization. By sweeping the $\alpha$ parameter and evaluation threshold $\varepsilon$, the experiment traces empirical G-I tradeoff curves and compares them against the theoretical Pareto front.

## Pipeline

```
CUB-200 images
    │
    ▼
ResNet-50 feature extraction (pretrained, fine-tuned)
    │
    ▼
Phylogenetic distance matrix (dendropy, birds_species.nwk)
    │
    ▼
Training with dual loss: (1-α)·L_ID + α·L_Gen
    │
    ▼
Threshold sweep: evaluate G-score and I-score at each ε
    │
    ▼
create_pickle_files.py → training_data.pkl, threshold_data.pkl
    │
    ▼
figure.ipynb / supp.ipynb → paper figures
```

## Prerequisites

1. **CUB-200-2011 dataset**: Download from [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract to `CNN_birds/CUB_200_2011/`
2. **Conda environment**:
   ```bash
   conda env create -f CNN_birds/env.yml
   conda activate miller_law
   ```

## Step 1: Training

```bash
cd CNN_birds
python src/birdsbirdsbirds.py --alpha 0.0 0.25 0.5 0.75 0.8 0.85 0.9 0.95 1.0 \
                               --epochs 15 --num-seeds 5
```

This will:
1. Build the phylogenetic distance matrix from `data/birds_species.nwk` (cached after first run)
2. For each alpha value and seed, train a ResNet-50 with mixed-precision and gradient accumulation
3. Save model checkpoints to `data/models/{YYYYMMDD_HHMMSS}/model_alpha{A}_epoch{E}.pt`
4. Save threshold-sweep CSVs to `results/models_{YYYYMMDD_HHMMSS}/threshold_results_alpha{A}.csv`

### Training Details

- **Feature extraction**: ResNet-50 backbone pretrained on ImageNet, final FC layer replaced
- **Mixed precision**: Uses `torch.amp.GradScaler` for memory efficiency
- **Gradient accumulation**: Effective batch size = `batch_size * grad_accum` (default 8 * 4 = 32)
- **Dual loss**: Cross-entropy for species ID, MSE to phylogenetic distance target for generalization
- **Threshold evaluation**: At each epoch end, sweeps $\varepsilon$ from 0 to 1 in 50 steps, computing G-score and I-score at each threshold

### Memory Considerations

For GPUs with limited memory:
```bash
python src/birdsbirdsbirds.py --batch-size 4 --grad-accum 8  # same effective batch, less memory
```

## Step 2: Aggregation

After training completes for all alpha values and seeds:

1. Note the timestamps of your training runs (directory names in `data/models/`)
2. Edit `src/create_pickle_files.py`, updating `DATE_PATTERNS`:
   ```python
   DATE_PATTERNS = ["20250513_*", "20250514_*"]  # your timestamps here
   ```
3. Run:
   ```bash
   python src/create_pickle_files.py
   ```

This produces:
- `results/output/training_data.pkl` — `{alpha -> {seed -> {epoch -> {g_score, i_score, ood_g_score}}}}`
- `results/output/threshold_data.pkl` — `{alpha -> {threshold -> {i_score: [...], g_score: [...], ...}}}`
- `results/output/evolutionary_similarity.pkl` — normalized inverse-distance similarity matrix
- `results/output/evolutionary_similarity_matrix.png` — heatmap visualization

## Step 3: Figures

Open and run the notebooks:
- **`notebooks/figure.ipynb`** — Main paper figures (G-I tradeoff curves, Pareto fronts, alpha sweep)
- **`notebooks/supp.ipynb`** — Supplementary analyses (training dynamics, per-species breakdown)

Expected data paths (relative to notebook working directory):
- `../src/training_data.pkl`
- `../results/output/threshold_data.pkl`
- `../data/evo_distance_matrix.npy` (generated during training)
- `../results/supplementary/training_dynamics_raw.pkl`

## Key Functions in birdsbirdsbirds.py

| Function | Description |
|----------|-------------|
| `get_evolutionary_distance()` | Computes MRCA-based phylogenetic distance using dendropy |
| `normalize_distance_matrix()` | Scales distances to [0, 1] range |
| `theoretical_G_score(eps, alpha)` | Miller's Law analytical prediction for G-score |
| `theoretical_I_score(eps, alpha)` | Miller's Law analytical prediction for I-score |
| `train_with_gi_tradeoff()` | Main training loop with dual loss and evaluation |
| `evaluate_identification()` | Threshold-based identification evaluation |
| `evaluate_generalization()` | Threshold-based generalization evaluation |
| `run_experiments()` | Orchestrates runs across alpha values and seeds |

## Output File Formats

### Model Checkpoints (`.pt`)

PyTorch checkpoint dicts containing:
- `model_state_dict` — model weights
- `g_score` — generalization score at save epoch
- `i_score` — identification score at save epoch
- `ood_g_score` — out-of-distribution generalization score
- `epoch` — epoch number

### Threshold Results CSV

One row per threshold value:

| Column | Description |
|--------|-------------|
| `threshold` | $\varepsilon$ value |
| `g_score` | Generalization accuracy at this threshold |
| `i_score` | Identification accuracy at this threshold |
| `ood_g_score` | OOD generalization accuracy |
