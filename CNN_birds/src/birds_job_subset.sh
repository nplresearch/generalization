#!/bin/bash
#SBATCH --job-name=birds_gi          # Job name
#SBATCH --output=./logs/birds_alpha_%A_%a.log  # Output file (%A = job ID, %a = array index)
#SBATCH --error=./logs/birds_alpha_%A_%a.err   # Error file
#SBATCH --partition=gpu              # GPU partition 
#SBATCH --gres=gpu:v100-pcie:1       # Request 1 GPU
#SBATCH --time=04:00:00              # Time limit 8 hours
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=64G                    # Memory
#SBATCH --array=0-3                  # CUSTOMIZE THIS for number of alpha values

# Define which alpha values to run (customize this array)
# Example: first two alphas (0.0 and 0.25)
# ALPHAS=(0.0 0.25)

# Example: middle alphas
# ALPHAS=(0.5 0.75 0.8)

# Example: high alphas
ALPHAS=(0.85 0.9 0.95 1.0)

# Get the alpha value for this array job
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
echo "This job will run with alpha = $ALPHA"

# Print some diagnostics
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Array job ID: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
nvidia-smi

# Load any modules your cluster requires
# module load cuda/12.3  # Uncomment if your cluster uses environment modules

# Activate your Python environment (update paths as needed)
# Option 1: If you have a conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh  # Adjust path as needed
conda activate miller_law                        # Your environment name

# Option 2: If you have a venv
# source $HOME/venvs/torch39/bin/activate        # Adjust path as needed

# Set environment variable to prevent PyTorch from downloading pretrained models
export TORCH_HOME=$HOME/.torch  # Ensure this directory exists and has any models you need
export PYTORCH_PRETRAINED_BERT_CACHE=$HOME/.pytorch_pretrained_bert

# Switch to the working directory
cd /work/JeFeSpace/generalization_transformer

# Make sure we're in the right directory
echo "Working directory: $(pwd)"
ls -l

# Add to your slurm script (before running python)
if nvidia-smi | grep -q "K40"; then
    echo "WARNING: Using CPU mode due to K40 GPU detection"
    export CUDA_VISIBLE_DEVICES=""
fi

# Make the logs directory if it doesn't exist
mkdir -p ./logs

# Run the script with the specific alpha value
echo "Starting script execution with alpha=$ALPHA..."
python src/birdsbirdsbirds_v2.py --alpha $ALPHA --batch-size 32 --grad-accum 4

# Report completion
echo "Job finished on $(date)" 