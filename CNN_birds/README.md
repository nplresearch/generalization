# Birds Species Generalization-Identification Experiments

This repository contains code for evaluating the trade-off between generalization and identification in embedding spaces using bird species data.

## Overview

The `birdsbirdsbirds.py` script trains neural networks with varying degrees of evolutionary distance alignment to study how embedding spaces can balance the ability to identify individual species while maintaining generalization capabilities that reflect evolutionary relationships.

## Prerequisites

- Python 3.7+
- PyTorch with CUDA support (recommended)
- Required packages are specified in the `env.yml` file
- CUB_200_2011 dataset in the parent directory of the script

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd generalization
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f env.yml
   ```

3. Download and extract the CUB_200_2011 dataset to the parent directory

## Basic Usage

```bash
cd generalization/src
python birdsbirdsbirds.py
```

## Command Line Arguments

The script supports several command line arguments:

- `--alpha`: Control the trade-off between identification and generalization (0.0-1.0)
  - Lower values (0.0) prioritize species identification
  - Higher values (1.0) prioritize evolutionary distance alignment
  - Default: 0.5

- `--batch-size`: Number of samples per batch (default: 8)
- `--grad-accum`: Gradient accumulation steps for larger effective batch sizes (default: 4)
- `--epochs`: Number of training epochs (default: 15)
- `--num-seeds`: Number of random seeds to use (default: 5)
- `--seeds`: Specific random seeds to use (overrides num-seeds)

## Example Commands

### Train with default settings:
```bash
python birdsbirdsbirds.py
```

### Train with focus on identification:
```bash
python birdsbirdsbirds.py --alpha 0.0
```

### Train with focus on generalization:
```bash
python birdsbirdsbirds.py --alpha 1.0
```

### Train with balanced approach and longer training:
```bash
python birdsbirdsbirds.py --alpha 0.5 --epochs 30
```

### Run multiple experiments with different alpha values:
```bash
python birdsbirdsbirds.py --alpha 0.0 0.25 0.5 0.75 1.0
```

### Specify memory-efficient settings for large models:
```bash
python birdsbirdsbirds.py --batch-size 4 --grad-accum 8
```

## Output

- Models are saved in `../data/models/{timestamp}/`
- Results and plots are saved in `../results/models_{timestamp}/`
- Key outputs include:
  - Saved model checkpoints for each epoch
  - G-I tradeoff plots
  - CSV files with performance metrics for different thresholds
  - Training dynamic plots if multiple alpha values are used

## Implementation Details

The script implements:
- Evolutionary distance calculation between bird species
- Threshold-based similarity judgments for generalization and identification
- Triplet-based loss function that aligns embedding distances with evolutionary distances
- Evaluation across multiple thresholds to analyze the G-I tradeoff

## Running the Analysis Pipeline

To reproduce the results, follow these steps:

1. Train models with different alpha values using `birdsbirdsbirds.py`
2. Generate pickle files for analysis using `create_pickle_files.py` 
3. Run the analysis notebooks:
   - `figure.ipynb` - Generates the main figures
   - `supp.ipynb` - Generates supplementary analyses

### Important Notes

- Before running `create_pickle_files.py`, update the `DATE_PATTERNS` variable with your specific training run timestamps. The default is set to:
  ```python
  DATE_PATTERNS = ["20250513_*", "20250514_*"]
  ```
  Replace these with the actual dates of your training runs (found in the directory names under `data/models/`).

- The notebook files expect specific paths for data files. If you experience issues with file paths, check that:
  - The evolutionary distance matrix is at `../data/evo_distance_matrix.npy`
  - The training data is accessible at `../src/training_data.pkl` and `../results/output/threshold_data.pkl`
  - The supplementary data is at `../results/supplementary/training_dynamics_raw.pkl`

## Notes

- The script automatically uses GPU if available
- For low-memory GPUs, use smaller batch sizes with higher gradient accumulation
- First run may take longer as it builds the evolutionary distance matrix
