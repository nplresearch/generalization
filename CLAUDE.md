# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository investigates the **Generalization-Identification (G-I) tradeoff** predicted by Miller's Law across three experimental paradigms:

1. **CNN_birds/** — CNNs trained on CUB-200 bird species dataset, using phylogenetic distance as ground truth for generalization
2. **Toy_model/** — Synthetic experiments with controlled distance metrics validating Miller's Law predictions
3. **VLM_shapes/** — Vision Language Model embeddings tested on color, number, and position similarity/identification
4. **LLM_dates/** — LLM analysis experiments with saved results

## Key Concepts

- **α parameter** (0.0–1.0): Controls training emphasis. α=0 is pure identification, α=1 is pure generalization. Loss = `(1-α) × ID_loss + α × Gen_loss`
- **Threshold ε**: Similarity decisions based on feature distance ≤ ε. Varying ε traces the G-I tradeoff curve
- **G score / I score**: Generalization and identification metrics evaluated at each threshold

## Environment Setup

```bash
conda env create -f CNN_birds/configs/env.yml
conda activate miller_law
```

Key dependencies: Python 3.9, PyTorch 2.1, dendropy (phylogenetic trees), wandb, torchvision, sklearn.

## Running CNN Experiments

```bash
# Train models across alpha values (from CNN_birds/)
python src/birdsbirdsbirds.py --alpha 0.5 --epochs 15 --num-seeds 5

# Key CLI arguments:
#   --alpha        G-I tradeoff weight (0.0 to 1.0)
#   --batch-size   Training batch size (default: 8)
#   --grad-accum   Gradient accumulation steps (default: 4)
#   --seeds        Specific random seeds (overrides --num-seeds)
```

### Analysis Pipeline

1. Train models with different alpha values via `birdsbirdsbirds.py`
2. Aggregate results: `python src/create_pickle_files.py` (edit `DATE_PATTERNS` to match your run dates)
3. Generate figures: run `CNN_birds/notebooks/figure.ipynb` and `supp.ipynb`

## Architecture

### CNN_birds/src/birdsbirdsbirds.py (~1800 lines, monolithic)

Core training script. Key functions:
- `get_evolutionary_distance()` — MRCA-based phylogenetic distance via DendriPy
- `train_with_gi_tradeoff()` — Main training loop with dual loss
- `evaluate_identification()` / `evaluate_generalization()` — Threshold-based evaluation
- `theoretical_G_score()` / `theoretical_I_score()` — Miller's Law analytical predictions
- `run_experiments()` — Orchestrates runs across alpha values and seeds

Data flow: CUB-200 images → ResNet50 features → distance matrix → α-weighted training → threshold evaluation → G-I curves vs. theory

### Toy_model/src/

Three modules implementing controlled synthetic experiments:
- **MillerDataGen.py** — `MillerDataGenerator` (standard) and `CompositionalDataGenerator` (multi-slot) produce similarity/identification test batches
- **MillerTrainer.py** — Training loop with similarity + reconstruction loss, LR scheduling
- **MillerTest.py** — `TransformerModel` (linear projection) and `CompositionalModel` (slot-wise embeddings); evaluation via `MillerTestEvaluator`

Notebooks in `Toy_model/notebooks/` run the slot vs. non-slot comparisons.

### VLM_shapes/ and LLM_dates/

Jupyter notebook-driven experiments with pre-computed results in `data/` and `saved_results/` directories. No separate Python modules—analysis is self-contained in notebooks.

## Configuration

- `CNN_birds/configs/default.yaml` — Model hyperparameters (hidden_dim, learning_rate, etc.)
- `CNN_birds/configs/env.yml` — Conda environment specification
- `create_pickle_files.py` uses hardcoded `DATE_PATTERNS` and `ALPHA_VALUES` lists to find and aggregate experiment outputs
