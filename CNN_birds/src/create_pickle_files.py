"""
Post-training aggregation of CNN experiment results into pickle files.

After birdsbirdsbirds.py has been run for multiple α values and random seeds,
this script scans the timestamped output directories, collects per-epoch
metrics and threshold-sweep results, and serialises them into pickle files
that the analysis notebooks (figure.ipynb, supp.ipynb) consume.

**Important**: update DATE_PATTERNS to match the timestamps of your training
runs before executing.

Outputs (written to ../results/output/):
  - training_data.pkl:  {α → {run_id → {epoch → {g_score, i_score, ...}}}}
  - threshold_data.pkl: {α → {threshold → {i_score: [...], g_score: [...], ...}}}
  - evolutionary_similarity.pkl: normalised inverse-distance similarity matrix
  - evolutionary_similarity_matrix.png: heatmap visualisation
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set up paths
BASE_MODEL_DIR = "../data/models"
BASE_RESULTS_DIR = "../results"
OUTPUT_DIR = "../results/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the alpha values we're interested in
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Define date patterns to filter model directories
DATE_PATTERNS = ["20250513_*", "20250514_*"] #UPDATE THIS AS NEEDED

# Function to load model data directly from the .pt file
def load_model_metrics(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        return {
            'g_score': checkpoint.get('g_score', 0),
            'i_score': checkpoint.get('i_score', 0),
            'ood_g_score': checkpoint.get('ood_g_score', 0),
            'epoch': int(model_path.split('epoch')[-1].split('.')[0])
        }
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

# Function to collect model data for training trajectories
def collect_training_trajectories():
    print("Collecting training trajectories data...")
    # Structure: alpha -> seed -> epoch -> metrics
    data = defaultdict(lambda: defaultdict(dict))
    
    for date_pattern in DATE_PATTERNS:
        # Find all model directories matching the date pattern
        model_dirs = glob.glob(os.path.join(BASE_MODEL_DIR, date_pattern))
        
        for model_dir in model_dirs:
            # Extract seed/run from directory name
            seed = os.path.basename(model_dir)
            
            # Find all model files in this directory
            model_files = glob.glob(os.path.join(model_dir, "model_alpha*.pt"))
            
            for model_file in model_files:
                # Extract alpha from filename
                alpha_str = model_file.split("model_alpha")[1].split("_")[0]
                try:
                    alpha = float(alpha_str)
                    
                    if alpha in ALPHA_VALUES:
                        # Load metrics from checkpoint
                        metrics = load_model_metrics(model_file)
                        if metrics:
                            data[alpha][seed][metrics['epoch']] = metrics
                except ValueError:
                    print(f"Skipping file with invalid alpha: {model_file}")
    
    return data

# Function to collect threshold data for G-I tradeoff plots
def collect_threshold_data():
    print("Collecting threshold data...")
    # Structure: threshold -> alpha -> seed -> metrics
    data = defaultdict(lambda: defaultdict(list))
    # Structure for final metrics: alpha -> threshold -> [list of metrics across seeds]
    final_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for date_pattern in DATE_PATTERNS:
        # Find all results directories matching the date pattern
        results_dirs = glob.glob(os.path.join(BASE_RESULTS_DIR, f"models_{date_pattern}"))
        
        for results_dir in results_dirs:
            # Extract seed/run from directory name
            seed = os.path.basename(results_dir).replace("models_", "")
            
            # Find all threshold results files in this directory
            csv_files = glob.glob(os.path.join(results_dir, "threshold_results_alpha*.csv"))
            
            for csv_file in csv_files:
                # Extract alpha from filename
                alpha_str = csv_file.split("threshold_results_alpha")[1].split(".csv")[0]
                try:
                    alpha = float(alpha_str)
                    
                    if alpha in ALPHA_VALUES:
                        try:
                            # Load CSV data
                            df = pd.read_csv(csv_file)
                            
                            # Store data by threshold->alpha->seed
                            for _, row in df.iterrows():
                                threshold = row['threshold']
                                final_data[alpha][threshold]['i_score'].append(row['i_score'])
                                final_data[alpha][threshold]['g_score'].append(row['g_score'])
                                final_data[alpha][threshold]['ood_g_score'].append(row['ood_g_score'])
                                
                                # We're not using theoretical values for plotting
                        except Exception as e:
                            print(f"Error processing {csv_file}: {e}")
                except ValueError:
                    print(f"Skipping file with invalid alpha: {csv_file}")
    
    return final_data

# Function to create evolutionary similarity matrix
def create_evolutionary_similarity():
    print("Creating evolutionary similarity matrix...")
    
    # First, load the evolutionary distance matrix
    try:
        evo_distances = np.load("../data/evo_distance_matrix.npy")
        print(f"Loaded evolutionary distance matrix with shape {evo_distances.shape}")
    except Exception as e:
        print(f"Error loading evolutionary distance matrix: {e}")
        return None
    
    # Convert evolutionary distances to similarity (invert and normalize)
    # High similarity = low distance
    evo_similarity = 1 / (1 + evo_distances)
    
    # Normalize the matrix
    evo_similarity = (evo_similarity - evo_similarity.min()) / (evo_similarity.max() - evo_similarity.min())
    
    return evo_similarity

# Save collected data using pickle
def save_data(data, filename):
    # Convert nested defaultdicts to regular dictionaries
    def convert_defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
        return d
    
    # Convert the data structure
    regular_dict_data = convert_defaultdict_to_dict(data)
    
    # Save using pickle
    with open(filename, 'wb') as f:
        pickle.dump(regular_dict_data, f)
    print(f"Data saved to {filename}")

def main():
    # Create training data pickle
    training_data = collect_training_trajectories()
    
    # Save to current directory (for compatibility with original path)
    save_data(training_data, 'training_data.pkl')
    
    # Also save to output directory
    save_data(training_data, os.path.join(OUTPUT_DIR, 'training_data.pkl'))
    
    # Create threshold data pickle
    threshold_data = collect_threshold_data()
    
    # Save to current directory (for compatibility with original path)
    save_data(threshold_data, 'threshold_data.pkl')
    
    # Also save to output directory
    save_data(threshold_data, os.path.join(OUTPUT_DIR, 'threshold_data.pkl'))
    
    # Create evolutionary similarity matrix
    evo_similarity = create_evolutionary_similarity()
    
    if evo_similarity is not None:
        # Save the evolutionary similarity matrix
        with open(os.path.join(OUTPUT_DIR, "evolutionary_similarity.pkl"), 'wb') as f:
            pickle.dump(evo_similarity, f)
        print(f"Evolutionary similarity saved to {os.path.join(OUTPUT_DIR, 'evolutionary_similarity.pkl')}")
        
        # Create a visualization of the evolutionary similarity matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(evo_similarity, cmap='magma')
        plt.title('Evolutionary Similarity Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'evolutionary_similarity_matrix.png'), dpi=300)
        plt.close()
    
    print("All pickle files created successfully!")

if __name__ == "__main__":
    main() 