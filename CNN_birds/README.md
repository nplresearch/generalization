# Birds Species Generalization-Identification Experiments

This repository contains code for evaluating the trade-off between generalization and identification in embedding spaces using bird species data.

## Overview

The `birdsbirdsbirds_v2.py` script trains neural networks with varying degrees of evolutionary distance alignment to study how embedding spaces can balance the ability to identify individual species while maintaining generalization capabilities that reflect evolutionary relationships.

## Prerequisites

- Python 3.7+
- PyTorch with CUDA support (recommended)
- Required packages are specified in the `env.yml` file
- CUB_200_2011 dataset in the parent directory of the script

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd generalization_transformer
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f env.yml
   ```

3. Download and extract the CUB_200_2011 dataset to the parent directory

## Basic Usage

```bash
cd generalization_transformer/src
python birdsbirdsbirds_v2.py
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
python birdsbirdsbirds_v2.py
```

### Train with focus on identification:
```bash
python birdsbirdsbirds_v2.py --alpha 0.0
```

### Train with focus on generalization:
```bash
python birdsbirdsbirds_v2.py --alpha 1.0
```

### Train with balanced approach and longer training:
```bash
python birdsbirdsbirds_v2.py --alpha 0.5 --epochs 30
```

### Run multiple experiments with different alpha values:
```bash
python birdsbirdsbirds_v2.py --alpha 0.0 0.25 0.5 0.75 1.0
```

### Specify memory-efficient settings for large models:
```bash
python birdsbirdsbirds_v2.py --batch-size 4 --grad-accum 8
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

## Notes

- The script automatically uses GPU if available
- For low-memory GPUs, use smaller batch sizes with higher gradient accumulation
- First run may take longer as it builds the evolutionary distance matrix
