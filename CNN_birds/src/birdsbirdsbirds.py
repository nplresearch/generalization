import dendropy
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import argparse  # Add argparse for command line arguments

# Add memory configuration for PyTorch CUDA allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Force matplotlib to use Agg backend which doesn't require GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from scipy.interpolate import make_interp_spline
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import datetime
import gc  # Add garbage collection

# Enhanced memory management function
def free_memory():
    """Free up GPU memory aggressively"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Define global device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Add diagnostic function to track progress 
def log_step(message):
    """Print timestamped log message"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Enhanced GPU memory monitoring
def print_gpu_memory_usage(label=""):
    """Print current GPU memory usage with an optional label"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = (total - used) / 1024**2
        print(f"GPU Memory [{label}]: Used {used:.1f}MB | Cached {cached:.1f}MB | Free {free:.1f}MB | Total {total:.1f}MB")

# Print initial memory status
print_gpu_memory_usage("Initial")

log_step("Starting script execution")

# Load and parse the Newick tree file
log_step("Loading Newick tree file...")
tree = dendropy.Tree.get(path="../data/birds_species.nwk", schema="newick")
log_step("Tree file loaded successfully")

# Function to calculate evolutionary distance between two species
def get_evolutionary_distance(tree, species1, species2):
    # Find the taxa for the species
    taxon1 = None
    taxon2 = None
    
    # Find taxon objects
    for taxon in tree.taxon_namespace:
        if taxon.label == species1:
            taxon1 = taxon
        if taxon.label == species2:
            taxon2 = taxon
    
    if taxon1 is None or taxon2 is None:
        return None
    
    # Get the leaf nodes that correspond to these taxa
    leaf1 = None
    leaf2 = None
    
    # Find leaf nodes by iterating through all tree nodes
    for node in tree.leaf_node_iter():
        if node.taxon == taxon1:
            leaf1 = node
        if node.taxon == taxon2:
            leaf2 = node
    
    if leaf1 is None or leaf2 is None:
        return None
    
    # Calculate path-length distance - sum branch lengths from each leaf to MRCA
    # Find most recent common ancestor (MRCA)
    try:
        mrca = tree.mrca(taxa=[taxon1, taxon2])
        if mrca is None:
            print(f"Warning: Could not find MRCA for {species1} and {species2}")
            return 0.0  # Return a default value to avoid breaking the script
        
        # Calculate distance from each leaf to MRCA
        dist1 = 0
        node = leaf1
        while node is not mrca and node is not None:
            if node.edge.length is not None:
                dist1 += node.edge.length
            node = node.parent_node
        
        dist2 = 0
        node = leaf2
        while node is not mrca and node is not None:
            if node.edge.length is not None:
                dist2 += node.edge.length
            node = node.parent_node
        
        # Total distance is the sum
        return dist1 + dist2
    except Exception as e:
        print(f"Error calculating distance between {species1} and {species2}: {e}")
        return 0.0  # Return a default value

# 3. Load CUB dataset and build mapping automatically
log_step("Loading CUB dataset...")
cub_root = '../CUB_200_2011'
images_dir = os.path.join(cub_root, 'images')

# Read class names
with open(os.path.join(cub_root, 'classes.txt'), 'r') as f:
    class_names = [line.split('.')[1].strip() for line in f.readlines()]

# Read scientific names from the tree
scientific_names = set()
for taxon in tree.taxon_namespace:
    if taxon.label:
        scientific_names.add(taxon.label)

print(f"Found {len(scientific_names)} scientific names in the tree")
print(f"Found {len(class_names)} common names in the CUB dataset")

# We'll try to match each common name word with scientific name words
common_to_scientific = {}

for common_name in class_names:
    # Clean common name (replace underscores with spaces)
    common_name_clean = common_name.replace('_', ' ')
    
    # Split the common name into words
    common_words = common_name_clean.lower().split()
    
    # Try to find matching scientific names
    potential_matches = []
    
    for sci_name in scientific_names:
        # Split scientific name - normally has genus and species
        sci_parts = sci_name.lower().split()
        
        # Check for match - we'll use a more sophisticated scoring approach
        match_score = 0
        
        # Check if genus/species words appear in common name
        for sci_part in sci_parts:
            if len(sci_part) >= 5:  # Only use meaningful parts
                for word in common_words:
                    if len(word) >= 5:  # Avoid short words like "the", "and", etc.
                        # Exact match gets highest score
                        if sci_part == word:
                            match_score += 3
                        # Partial matches get lower scores
                        elif sci_part.startswith(word) or word.startswith(sci_part):
                            match_score += 2
                        elif sci_part in word or word in sci_part:
                            match_score += 1
        
        if match_score > 0:
            potential_matches.append((sci_name, match_score))
    
    # Sort by match score (highest first)
    potential_matches.sort(key=lambda x: x[1], reverse=True)
    
    # If we have potential matches, use the best one
    if potential_matches:
        common_to_scientific[common_name] = potential_matches[0][0]

print(f"Automatically mapped {len(common_to_scientific)} species")

# After the automated mapping, let's add some specific, verified mappings
manual_mappings = {
    'Black_footed_Albatross': 'Phoebastria nigripes',
    'Laysan_Albatross': 'Phoebastria immutabilis',
    'Sooty_Albatross': 'Phoebetria fusca',
    'Groove_billed_Ani': 'Crotophaga sulcirostris',
    'Crested_Auklet': 'Aethia cristatella',
    'Least_Auklet': 'Aethia pusilla',
    'Parakeet_Auklet': 'Aethia psittacula',
    'Rhinoceros_Auklet': 'Cerorhinca monocerata',
    'Brewer_Blackbird': 'Euphagus cyanocephalus',
    'Red_winged_Blackbird': 'Agelaius phoeniceus',
    'Rusty_Blackbird': 'Euphagus carolinus',
    'Yellow_headed_Blackbird': 'Xanthocephalus xanthocephalus',
    'Bobolink': 'Dolichonyx oryzivorus',
    'Indigo_Bunting': 'Passerina cyanea',
    'Lazuli_Bunting': 'Passerina amoena',
    'Painted_Bunting': 'Passerina ciris',
    'Cardinal': 'Cardinalis cardinalis',
}

# Override with manual mappings where available
for common, sci in manual_mappings.items():
    if common in class_names and common_to_scientific.get(common) != sci:
        # Check if the scientific name is in the tree
        found = False
        for taxon in tree.taxon_namespace:
            if taxon.label == sci:
                found = True
                break
        
        if found:
            common_to_scientific[common] = sci
            print(f"Manual override: {common} -> {sci}")

print(f"Final mapping contains {len(common_to_scientific)} species")
for common, sci in list(common_to_scientific.items())[:5]:  # Show first 5 mappings
    print(f"  {common} -> {sci}")

# Validate all mappings to make sure each species exists in the tree
validated_mappings = {}
for common, sci in common_to_scientific.items():
    # Check if the scientific name is in the tree
    found = False
    for taxon in tree.taxon_namespace:
        if taxon.label == sci:
            found = True
            break
    
    if found:
        validated_mappings[common] = sci
    else:
        print(f"Warning: Could not find {sci} in the tree for {common}")

# Replace the original mapping with the validated one
common_to_scientific = validated_mappings
print(f"Validated mapping contains {len(common_to_scientific)} species")

# Find species that are in both datasets
available_species = []
for species in class_names:
    if species in common_to_scientific:
        scientific_name = common_to_scientific[species]
        if tree.find_node_with_taxon_label(scientific_name) is not None:
            available_species.append(species)

print(f"Found {len(available_species)} species in both datasets that can be used")

# Build the evolutionary distance matrix
log_step("Building evolutionary distance matrix...")
species_count = len(available_species)
evo_distances = np.zeros((species_count, species_count))

for i, sp1 in enumerate(available_species):
    for j, sp2 in enumerate(available_species):
        if i == j:
            continue
        sci1 = common_to_scientific[sp1]
        sci2 = common_to_scientific[sp2]
        distance = get_evolutionary_distance(tree, sci1, sci2)
        if distance is not None:
            evo_distances[i, j] = distance
log_step("Evolutionary distance matrix built")

# save the evolutionary distance matrix
np.save("../data/evo_distance_matrix.npy", evo_distances)

# Load images and extract features
# Use a pre-trained CNN to extract features
log_step("Loading ResNet50 model (this may take some time)...")
# IMPORTANT CHANGE: Use a locally initialized model without downloading weights
try:
    # First try loading the model with pretrained weights from local cache (if it exists)
    weights_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
    if os.path.exists(weights_path):
        log_step(f"Found local weights at {weights_path}")
        # Load model without downloading, will use local weights
        model = models.resnet50()
        model.load_state_dict(torch.load(weights_path))
        log_step("Loaded ResNet50 with pretrained weights from local cache")
    else:
        # Initialize with random weights - no internet download
        log_step("No pretrained weights found locally, initializing with random weights")
        model = models.resnet50(pretrained=False)
        log_step("Initialized ResNet50 with random weights")
except Exception as e:
    log_step(f"Error during model loading: {e}")
    log_step("Falling back to random initialization")
    model = models.resnet50(pretrained=False)

log_step("ResNet50 model loaded successfully")

# Move model to correct device after loading
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
log_step("Feature extractor created")
feature_extractor = feature_extractor.to(device)  # Move to GPU if available
log_step(f"Feature extractor moved to {device}")
feature_extractor.eval()
log_step("Feature extractor ready for inference")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
log_step("Image transforms created")

# Extract features for each species
log_step("Starting feature extraction for each species...")
features_by_species = {}
for i, species in enumerate(available_species):
    # Find image directory for this species
    species_idx = class_names.index(species) + 1  # CUB uses 1-indexed classes
    species_dir = os.path.join(images_dir, f"{species_idx:03d}.{species}")
    
    # Load up to 10 images for this species
    species_features = []
    img_paths = [os.path.join(species_dir, img_file) for img_file in os.listdir(species_dir)[:10]]
    
    # Log progress every 10 species
    if i % 10 == 0:
        log_step(f"Processing species {i+1}/{len(available_species)}: {species}")
    
    # Process in small batches to better utilize GPU
    batch_size = 5  # Adjust based on GPU memory
    for j in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[j:j+batch_size]
        batch_tensors = []
        
        try:
            # Prepare batch
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                batch_tensors.append(img_tensor)
            
            if batch_tensors:
                # Stack tensors and process as a batch
                batch = torch.cat(batch_tensors, dim=0).to(device)
                
                with torch.no_grad():
                    batch_features = feature_extractor(batch).squeeze().cpu().numpy()
                    # Handle case where batch size is 1
                    if len(batch_features.shape) == 1:
                        batch_features = batch_features.reshape(1, -1)
                    
                    for feat in batch_features:
                        species_features.append(feat.flatten())
        except Exception as e:
            print(f"Error processing batch for {species}: {e}")
    
    if species_features:
        features_by_species[species] = np.array(species_features)
        print(f"Processed {len(species_features)} images for {species}")

# ===================== NEW FUNCTIONS FOR MILLER'S LAW ANALYSIS =====================

def calculate_average_ball_measure(threshold, distances_matrix):
    """
    Calculate average measure of a ball with radius threshold.
    
    For each point, compute the fraction of points within threshold distance,
    then return the average and variance across all points.
    
    Args:
        threshold: Ball radius (epsilon value)
        distances_matrix: Matrix of distances between points
    
    Returns:
        avg_ball: Average measure of balls with radius threshold
        var_ball: Variance of ball measures
    """
    ball_measures = []
    n = distances_matrix.shape[0]
    
    for i in range(n):
        # Count points within threshold distance (including the point itself)
        ball_size = np.sum(distances_matrix[i] <= threshold)
        ball_measure = ball_size / n  # Normalize
        ball_measures.append(ball_measure)
    
    avg_ball = np.mean(ball_measures)
    var_ball = np.var(ball_measures)
    
    return avg_ball, var_ball

def calculate_alpha_term(distances_matrix, epsilon=1e-5):
    """
    Calculate the α term for non-discriminative spaces.
    
    α = ∫₀^∞ ⟨σ(r)/2⟩dr
    
    For discrete spaces, approximated by summing over unique distances.
    
    Args:
        distances_matrix: Matrix of distances between points
        epsilon: Tolerance for considering points as being at the same distance
    
    Returns:
        alpha: The alpha term value
    """
    # Get unique distances (rounded to reduce near-duplicates)
    rounded_distances = np.round(distances_matrix.flatten(), int(-np.log10(epsilon)))
    unique_distances = np.unique(rounded_distances)
    
    alpha = 0
    n = distances_matrix.shape[0]
    
    for r in unique_distances:
        if r == 0:  # Skip zero distance (self distance)
            continue
            
        # Calculate average measure of spheres with radius r
        sphere_measures = []
        
        for i in range(n):
            # Count points at exactly distance r (or within a small epsilon)
            sphere_size = np.sum(np.abs(distances_matrix[i] - r) < epsilon)
            sphere_measure = sphere_size / n  # Normalize
            sphere_measures.append(sphere_measure)
        
        avg_sphere_measure = np.mean(sphere_measures)
        alpha += (avg_sphere_measure / 2)
    
    return alpha

def theoretical_G_score(threshold, distances_matrix):
    """
    Calculate theoretical G(ε) according to Theorem 1.
    
    G(ε) = (1/2)(1 + α) + ⟨b(ε)⟩ - ⟨b(ε)⟩² - Var(b(ε))
    
    Args:
        threshold: The threshold value (epsilon)
        distances_matrix: Matrix of distances between points
    
    Returns:
        G: Theoretical generalization score
    """
    avg_ball, var_ball = calculate_average_ball_measure(threshold, distances_matrix)
    alpha_term = calculate_alpha_term(distances_matrix)
    
    G = (1/2) * (1 + alpha_term) + avg_ball - avg_ball**2 - var_ball
    return G

def theoretical_I_score(threshold, distances_matrix):
    """
    Calculate theoretical I(ε) according to Theorem 2.
    
    I(ε) = 1 - (1/2)⟨b(ε)⟩
    
    Args:
        threshold: The threshold value (epsilon)
        distances_matrix: Matrix of distances between points
    
    Returns:
        I: Theoretical identification score
    """
    avg_ball, _ = calculate_average_ball_measure(threshold, distances_matrix)
    
    I = 1 - (1/2) * avg_ball
    return I

def is_discriminative(distances_matrix, epsilon=1e-5):
    """
    Check if the space is discriminative according to Definition 1.
    
    A space is discriminative if ⟨σ(r)⟩ = 0 ∀r ≥ 0,
    meaning no two points are exactly the same distance from a third point.
    
    Args:
        distances_matrix: Matrix of distances between points
        epsilon: Tolerance for considering points as being at the same distance
    
    Returns:
        bool: True if space is discriminative, False otherwise
    """
    # Get unique distances (rounded to reduce near-duplicates)
    rounded_distances = np.round(distances_matrix.flatten(), int(-np.log10(epsilon)))
    unique_distances = np.unique(rounded_distances)
    
    n = distances_matrix.shape[0]
    
    for r in unique_distances:
        if r == 0:  # Skip zero distance (self distance)
            continue
            
        # Check if there are any pairs of points at the same distance from a third point
        for i in range(n):
            # Find points at distance r from point i
            at_dist_r = np.where(np.abs(distances_matrix[i] - r) < epsilon)[0]
            
            if len(at_dist_r) >= 2:
                # Found at least two points at the same distance from point i
                return False
    
    return True

# Modified run_tests function to return both empirical and theoretical values
def run_tests(threshold, distances_matrix, n_trials=500):
    """
    Run empirical tests and calculate theoretical values for G-I tradeoff.
    
    Args:
        threshold: The threshold value (epsilon)
        distances_matrix: Matrix of distances between points
        n_trials: Number of trials for empirical tests
    
    Returns:
        g_empirical: Empirical generalization score
        i_empirical: Empirical identification score
        g_theoretical: Theoretical generalization score
        i_theoretical: Theoretical identification score
    """
    g_correct = 0
    g_total = 0
    i_correct = 0
    i_total = 0
    
    # Species indices for lookup
    species_indices = {sp: i for i, sp in enumerate(available_species)}
    
    # Similarity test
    for _ in range(n_trials):
        # Pick 3 random species that have features
        valid_species = [sp for sp in available_species if sp in features_by_species]
        species_sample = np.random.choice(valid_species, 3, replace=False)
        x1, x2, p = species_sample
        
        # Randomly pick images from each species
        x1_img = np.random.choice(len(features_by_species[x1]))
        x2_img = np.random.choice(len(features_by_species[x2]))
        p_img = np.random.choice(len(features_by_species[p]))
        
        # Get features
        x1_feat = features_by_species[x1][x1_img]
        x2_feat = features_by_species[x2][x2_img]
        p_feat = features_by_species[p][p_img]
        
        # Calculate visual distances
        d1 = np.linalg.norm(x1_feat - p_feat)
        d2 = np.linalg.norm(x2_feat - p_feat)
        
        # Ground truth (evolutionary distance)
        ed1 = distances_matrix[species_indices[x1], species_indices[p]]
        ed2 = distances_matrix[species_indices[x2], species_indices[p]]
        true_closer = x1 if ed1 < ed2 else x2
        
        
        # Apply threshold for similarity
        sim1 = 1 if d1 <= threshold else 0
        sim2 = 1 if d2 <= threshold else 0
        
        # Prediction based on similarity
        if sim1 == 0 and sim2 == 0:
            # Both outside threshold, random guess
            prediction = np.random.choice([x1, x2])
        else:
            # Based on similarity ratio
            p_x1 = sim1 / (sim1 + sim2) if (sim1 + sim2) > 0 else 0.5
            prediction = x1 if np.random.random() < p_x1 else x2
        
        if prediction == true_closer:
            g_correct += 1
        g_total += 1
    
    # Identification test
    for _ in range(n_trials):
        # Pick 2 random species
        valid_species = [sp for sp in available_species if sp in features_by_species]
        species_sample = np.random.choice(valid_species, 2, replace=False)
        x1, x2 = species_sample
        
        # Randomly pick images
        x1_img1 = np.random.choice(len(features_by_species[x1]))
        x1_img2 = np.random.choice(len(features_by_species[x1]))
        x2_img = np.random.choice(len(features_by_species[x2]))
        
        # Get features
        x1_feat1 = features_by_species[x1][x1_img1]
        x1_feat2 = features_by_species[x1][x1_img2]
        x2_feat = features_by_species[x2][x2_img]
        
        # Calculate distances
        d_same = np.linalg.norm(x1_feat1 - x1_feat2)
        d_diff = np.linalg.norm(x1_feat1 - x2_feat)
        
        # Apply threshold
        sim_same = 1 if d_same <= threshold else 0
        sim_diff = 1 if d_diff <= threshold else 0
        
        # Decision probability
        p_correct = sim_same / (sim_same + sim_diff) if (sim_same + sim_diff) > 0 else 0.5
        
        # Simulate decision
        if np.random.random() < p_correct:
            i_correct += 1
        i_total += 1
    
    # Calculate empirical scores
    g_empirical = g_correct/g_total if g_total > 0 else 0
    i_empirical = i_correct/i_total if i_total > 0 else 0
    
    # Calculate theoretical scores
    g_theoretical = theoretical_G_score(threshold, distances_matrix)
    i_theoretical = theoretical_I_score(threshold, distances_matrix)
    
    return g_empirical, i_empirical, g_theoretical, i_theoretical

# Run experiments with different thresholds
# First determine a reasonable range of thresholds based on feature distances
all_distances = []
for species, features in features_by_species.items():
    for i in range(min(5, len(features))):
        for other_species, other_features in features_by_species.items():
            for j in range(min(5, len(other_features))):
                dist = np.linalg.norm(features[i] - other_features[j])
                all_distances.append(dist)

# Check if distances were found
if len(all_distances) == 0:
    print("WARNING: No distances available for plotting - skipping visualization")

# Use the distribution of actual distances instead of artificial linear spacing
print(f"Collected {len(all_distances)} distances from the dataset")

# Sort and sample from the actual distances to create thresholds
# Round to reduce near-duplicates and remove exact duplicates
all_distances = np.array(sorted(list(set([round(d, 2) for d in all_distances]))))
print(f"After removing duplicates: {len(all_distances)} unique distances")

# Sample a manageable number of thresholds using percentile-based approach
# This gives more points in regions where distances are densely distributed
n_thresholds = 100  
if len(all_distances) > n_thresholds:
    percentiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.percentile(all_distances, percentiles)
    print(f"Sampled {len(thresholds)} thresholds using percentile sampling")
else:
    thresholds = all_distances
    print(f"Using all {len(thresholds)} unique distances as thresholds")

print(f"Threshold range: {thresholds.min():.2f} to {thresholds.max():.2f}")

# First, check if the space is discriminative
is_disc = is_discriminative(evo_distances)
print(f"Is the evolutionary distance space discriminative? {is_disc}")

# # Run tests and store results
# g_empirical_scores = []
# i_empirical_scores = []
# g_theoretical_scores = []
# i_theoretical_scores = []

# for threshold in thresholds:
#     g_emp, i_emp, g_theo, i_theo = run_tests(threshold, evo_distances, n_trials=2000)
#     g_empirical_scores.append(g_emp)
#     i_empirical_scores.append(i_emp)
#     g_theoretical_scores.append(g_theo)
#     i_theoretical_scores.append(i_theo)
#     print(f"Threshold: {threshold:.1f}, G_emp: {g_emp:.3f}, G_theo: {g_theo:.3f}, I_emp: {i_emp:.3f}, I_theo: {i_theo:.3f}")

# # Create a clean, professional plot with both empirical and theoretical curves
# if len(all_distances) > 0:  # Only try plotting if we have data
#     plt.figure(figsize=(10, 8))
    
#     # Create a color gradient based on threshold value
#     cmap = plt.cm.viridis
#     norm = plt.Normalize(thresholds.min(), thresholds.max())
#     colors = cmap(norm(thresholds))
    
#     # Plot empirical curves
#     plt.plot(g_empirical_scores, i_empirical_scores, '-', color='blue', 
#              linewidth=2.5, label='Empirical', alpha=0.8)
    
#     # Plot theoretical curves
#     plt.plot(g_theoretical_scores, i_theoretical_scores, '--', color='red', 
#              linewidth=2.5, label='Theoretical (Miller\'s Law)', alpha=0.8)
    
#     # Plot individual points with color gradient
#     for i in range(len(thresholds)):
#         plt.plot(g_empirical_scores[i], i_empirical_scores[i], 'o', color=colors[i], 
#                  markersize=6, markeredgecolor='white', markeredgewidth=1)

#     plt.xlabel('Generalization (G)', fontsize=14, fontweight='bold')
#     plt.ylabel('Identification (I)', fontsize=14, fontweight='bold')
#     plt.title('Generalization-Identification Tradeoff in Bird Species: Empirical vs Theoretical', 
#               fontsize=16, fontweight='bold', pad=20)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(loc='best', fontsize=12)

#     # Choose only a few points to annotate to avoid clutter
#     annotation_indices = np.linspace(0, len(thresholds)-1, 8, dtype=int)
#     for i in annotation_indices:
#         plt.annotate(f"ε={thresholds[i]:.1f}", 
#                    (g_empirical_scores[i], i_empirical_scores[i]),
#                    xytext=(10, 0), textcoords="offset points",
#                    fontsize=10, fontweight='bold',
#                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     ax = plt.gca()  # Get current axes reference
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('Threshold (ε)', fontsize=12, fontweight='bold')

#     plt.tight_layout()

#     plt.xlim([0.45, 0.85])
#     plt.ylim([0.45, 0.85])

#     plt.savefig('bird_g_i_tradeoff_with_theory.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Create a plot to analyze alpha and ball measure
#     plt.figure(figsize=(12, 5))
    
#     # Plot 1: Ball measure vs threshold
#     plt.subplot(1, 2, 1)
#     ball_measures = []
#     ball_variances = []
    
#     for threshold in thresholds:
#         avg_ball, var_ball = calculate_average_ball_measure(threshold, evo_distances)
#         ball_measures.append(avg_ball)
#         ball_variances.append(var_ball)
    
#     plt.plot(thresholds, ball_measures, 'b-', linewidth=2.5, label='Average ball measure')
#     plt.plot(thresholds, ball_variances, 'r--', linewidth=2.5, label='Ball measure variance')
#     plt.xlabel('Threshold (ε)', fontsize=12, fontweight='bold')
#     plt.ylabel('Measure', fontsize=12, fontweight='bold')
#     plt.title('Ball Measure vs. Threshold', fontsize=14, fontweight='bold')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(loc='best')
    
#     # Plot 2: Theoretical G scores with/without variance term
#     plt.subplot(1, 2, 2)
#     alpha_term = calculate_alpha_term(evo_distances)
#     g_no_var = [(1/2) * (1 + alpha_term) + ball_measures[i] - ball_measures[i]**2 
#                 for i, _ in enumerate(thresholds)]
    
#     plt.plot(thresholds, g_theoretical_scores, 'b-', linewidth=2.5, 
#              label='G(ε) with variance term')
#     plt.plot(thresholds, g_no_var, 'r--', linewidth=2.5, 
#              label='G(ε) without variance term')
#     plt.xlabel('Threshold (ε)', fontsize=12, fontweight='bold')
#     plt.ylabel('G(ε)', fontsize=12, fontweight='bold')
#     plt.title('Effect of Variance Term on G(ε)', fontsize=14, fontweight='bold')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(loc='best')
    
#     plt.tight_layout()
#     plt.savefig('ball_measure_analysis.png', dpi=300)
#     plt.close()
# else:
#     print("WARNING: No distances available for plotting - skipping visualization")

# Create data splitting functions for both in-distribution and OOD evaluation
def create_data_splits(available_species, test_size=0.2, ood_size=0.15, random_state=42):
    """
    Create data splits for both in-distribution and out-of-distribution evaluation
    
    Args:
        available_species: List of all available species
        test_size: Proportion of in-distribution test images
        ood_size: Proportion of species to hold out completely (OOD)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with all split information
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    
    # First, split species into those used for training and those held out (OOD)
    n_ood_species = max(1, int(len(available_species) * ood_size))
    train_species = random.sample(available_species, len(available_species) - n_ood_species)
    ood_species = [sp for sp in available_species if sp not in train_species]
    
    print(f"Training species: {len(train_species)}")
    print(f"Out-of-distribution species: {len(ood_species)}")
    
    # Now, for training species, split images into train/val/test
    image_splits = {}
    
    for species in available_species:
        if species not in features_by_species:
            continue
            
        # Get all image paths for this species
        species_idx = class_names.index(species) + 1
        species_dir = os.path.join(images_dir, f"{species_idx:03d}.{species}")
        image_files = os.listdir(species_dir)
        
        if species in train_species:
            # Split into train/val/test
            train_imgs, test_imgs = train_test_split(
                image_files, test_size=test_size, random_state=random_state)
            
            train_imgs, val_imgs = train_test_split(
                train_imgs, test_size=test_size, random_state=random_state)
            
            image_splits[species] = {
                'train': train_imgs,
                'val': val_imgs,
                'test': test_imgs,
                'split_type': 'in_distribution'
            }
        else:
            # OOD species - use all images for OOD testing
            image_splits[species] = {
                'train': [],
                'val': [],
                'test': image_files,
                'split_type': 'out_of_distribution'
            }
    
    return {
        'train_species': train_species,
        'ood_species': ood_species,
        'image_splits': image_splits
    }

# Create custom dataset that respects the data splits
class BirdDataset(Dataset):
    def __init__(self, species_list, features_by_species, evo_distances, species_indices, 
                 image_splits, split='train', transform=None):
        """
        Dataset for bird images with evolutionary distance information
        
        Args:
            species_list: List of all species
            features_by_species: Pre-extracted features by species
            evo_distances: Matrix of evolutionary distances
            species_indices: Mapping from species to indices
            image_splits: Dictionary with train/val/test split information
            split: Which split to use ('train', 'val', 'test', 'ood')
            transform: Image transformations to apply
        """
        self.samples = []
        self.transform = transform
        self.split = split
        
        # Set up species to index mapping (only for species in the current split)
        if split == 'ood':
            # For OOD, we need all species for reference
            relevant_species = species_list
        else:
            # For train/val/test, use only training species
            relevant_species = [sp for sp in species_list 
                                if sp in image_splits and image_splits[sp]['split_type'] == 'in_distribution']
            
        self.class_to_idx = {sp: i for i, sp in enumerate(relevant_species)}
        
        # Build dataset
        for species in species_list:
            if species not in features_by_species or species not in image_splits:
                continue
                
            # Skip species based on split
            if split == 'ood' and image_splits[species]['split_type'] != 'out_of_distribution':
                continue
            elif split != 'ood' and image_splits[species]['split_type'] != 'in_distribution':
                continue
                
            # Get images for this split
            img_files = image_splits[species][split if split != 'ood' else 'test']
            
            if not img_files:
                continue
                
            # Get path and prepare metadata
            species_idx = class_names.index(species) + 1
            species_dir = os.path.join(images_dir, f"{species_idx:03d}.{species}")
            
            for img_file in img_files:
                img_path = os.path.join(species_dir, img_file)
                
                # Store full evolutionary distance vector for this species
                evo_dists = evo_distances[species_indices[species]]
                
                # For OOD species, we don't have a class index, use -1
                class_idx = self.class_to_idx.get(species, -1)
                
                self.samples.append({
                    'img_path': img_path,
                    'species': species,
                    'label': class_idx,
                    'evo_distances': evo_dists
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image, 
            'label': sample['label'],
            'species': sample['species'],
            'evo_distances': sample['evo_distances']
        }

# At beginning after imports, add a memory monitoring function
def print_gpu_memory_usage(label=""):
    """Print current GPU memory usage with an optional label"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = (total - used) / 1024**2
        print(f"GPU Memory [{label}]: Used {used:.1f}MB | Cached {cached:.1f}MB | Free {free:.1f}MB | Total {total:.1f}MB")

# Model definition with feature extraction abilities
class GITradeoffModel(nn.Module):
    def __init__(self, num_classes, alpha=0.5, feature_dim=2048):
        super(GITradeoffModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace final classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original fc layer
        
        # Add a new classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Alpha controls the tradeoff between ID and generalization
        self.alpha = alpha
        
        # Print model size info for debugging
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params/1e6:.2f}M parameters")
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Get classification logits
        logits = self.classifier(features)
        
        return {
            'features': features,
            'logits': logits
        }
    
    def extract_features(self, x):
        """Extract feature representations only"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

# Custom evolutionary distance loss
class EvolutionaryDistanceLoss(nn.Module):
    def __init__(self, species_indices, evo_distances):
        super(EvolutionaryDistanceLoss, self).__init__()
        self.species_indices = species_indices
        # Store evolutionary distances as a tensor on the same device as the model
        if isinstance(evo_distances, np.ndarray):
            self.evo_distances = torch.from_numpy(evo_distances).float()
        else:
            self.evo_distances = evo_distances.float()
    
    def forward(self, features, species_batch):
        """
        Align feature distances with evolutionary distances
        """
        batch_size = features.size(0)
        device = features.device  # Get device from input features
        
        # Move evo_distances to the same device if needed
        if self.evo_distances.device != device:
            self.evo_distances = self.evo_distances.to(device)
        
        # Calculate pairwise distances in feature space
        feat_distances = torch.cdist(features, features, p=2)
        
        # Get corresponding evolutionary distances
        evo_dist_matrix = torch.zeros_like(feat_distances)
        
        # Convert species names to indices and look up evolutionary distances
        for i, sp1 in enumerate(species_batch):
            for j, sp2 in enumerate(species_batch):
                sp1_idx = self.species_indices[sp1]
                sp2_idx = self.species_indices[sp2] 
                evo_dist_matrix[i, j] = self.evo_distances[sp1_idx, sp2_idx]
        
        # Normalize both matrices to [0, 1] range
        if torch.max(feat_distances) > 0:
            feat_distances = feat_distances / torch.max(feat_distances)
        
        if torch.max(evo_dist_matrix) > 0:
            evo_dist_matrix = evo_dist_matrix / torch.max(evo_dist_matrix)
        
        # Calculate MSE between normalized matrices (upper triangular part)
        n = batch_size
        loss = torch.sum((feat_distances.triu(1) - evo_dist_matrix.triu(1)) ** 2) / (n * (n - 1) / 2)
        
        return loss

# Evaluation functions for in-distribution and OOD
def evaluate_identification(model, dataloader, device, threshold=15.0, n_trials=500):
    """
    Evaluate species identification accuracy (I-score) using a setup similar to 
    the generalization test with x1, x2, and a probe species p
    """
    model.eval()
    i_correct = 0
    i_total = 0
    
    # Get all species in the dataset
    all_samples = dataloader.dataset.samples
    species_to_samples = {}
    
    # Group samples by species
    for sample in all_samples:
        species = sample['species']
        if species not in species_to_samples:
            species_to_samples[species] = []
        species_to_samples[species].append(sample['img_path'])
    
    # Only keep species with at least 2 samples
    valid_species = [sp for sp, samples in species_to_samples.items() if len(samples) >= 2]
    
    if len(valid_species) < 2:
        print("Not enough valid species for identification test")
        return 0.0
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        # Run trials
        for _ in range(n_trials):
            # Pick 2 random species
            if len(valid_species) < 2:
                continue
                
            species_sample = np.random.choice(valid_species, 2, replace=False)
            x1, x2 = species_sample
            
            # Randomly select which species the probe comes from
            probe_species = np.random.choice([x1, x2])
            
            # Get random image paths
            x1_samples = species_to_samples[x1]
            x2_samples = species_to_samples[x2]
            probe_samples = species_to_samples[probe_species]
            
            if len(x1_samples) < 1 or len(x2_samples) < 1 or len(probe_samples) < 1:
                continue
                
            # Pick one image from each species and one from the probe
            x1_img_path = np.random.choice(x1_samples)
            x2_img_path = np.random.choice(x2_samples)
            
            # Make sure probe image is different from the one already selected 
            # for the matching species
            if probe_species == x1:
                remaining_samples = [s for s in probe_samples if s != x1_img_path]
                if not remaining_samples:
                    continue
                p_img_path = np.random.choice(remaining_samples)
            else:  # probe_species == x2
                remaining_samples = [s for s in probe_samples if s != x2_img_path]
                if not remaining_samples:
                    continue
                p_img_path = np.random.choice(remaining_samples)
            
            # Load and process images
            try:
                x1_img = transform(Image.open(x1_img_path).convert('RGB')).unsqueeze(0).to(device)
                x2_img = transform(Image.open(x2_img_path).convert('RGB')).unsqueeze(0).to(device)
                p_img = transform(Image.open(p_img_path).convert('RGB')).unsqueeze(0).to(device)
                
                # Extract features
                x1_feat = model.extract_features(x1_img).squeeze().cpu().numpy()
                x2_feat = model.extract_features(x2_img).squeeze().cpu().numpy()
                p_feat = model.extract_features(p_img).squeeze().cpu().numpy()
                
                # Calculate distances
                d1 = np.linalg.norm(x1_feat - p_feat)  # Distance between x1 and probe
                d2 = np.linalg.norm(x2_feat - p_feat)  # Distance between x2 and probe
                
                # Apply threshold for similarity
                sim1 = 1 if d1 <= threshold else 0
                sim2 = 1 if d2 <= threshold else 0
                
                # Prediction based on similarity
                if sim1 == 0 and sim2 == 0:
                    # Both outside threshold, random guess
                    prediction = np.random.choice([x1, x2])
                else:
                    # Based on similarity ratio
                    p_x1 = sim1 / (sim1 + sim2) if (sim1 + sim2) > 0 else 0.5
                    prediction = x1 if np.random.random() < p_x1 else x2
                
                # Check if prediction matches the true probe species
                if prediction == probe_species:
                    i_correct += 1
                i_total += 1
                
            except Exception as e:
                continue
    
    return i_correct / i_total if i_total > 0 else 0

def evaluate_generalization(model, threshold, species_indices, evo_distances, 
                            available_species, features_by_species, device, n_trials=500):
    """
    Evaluate generalization performance (G-score) using similarity test
    """
    model.eval()
    g_correct = 0
    g_total = 0
    
    # Convert evolutionary distances to tensor if it's numpy array
    if isinstance(evo_distances, np.ndarray):
        evo_distances_tensor = torch.from_numpy(evo_distances).float().to(device)
    else:
        evo_distances_tensor = evo_distances.float().to(device)
    
    with torch.no_grad():
        # Similarity test
        for _ in range(n_trials):
            # Pick 3 random species that have features
            valid_species = [sp for sp in available_species if sp in features_by_species]
            if len(valid_species) < 3:
                continue
                
            species_sample = np.random.choice(valid_species, 3, replace=False)
            x1, x2, p = species_sample
            
            # Load and process images
            x1_img_path = get_random_image_path(x1)
            x2_img_path = get_random_image_path(x2)
            p_img_path = get_random_image_path(p)
            
            if not x1_img_path or not x2_img_path or not p_img_path:
                continue
                
            # Process images
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            x1_img = transform(Image.open(x1_img_path).convert('RGB')).unsqueeze(0).to(device)
            x2_img = transform(Image.open(x2_img_path).convert('RGB')).unsqueeze(0).to(device)
            p_img = transform(Image.open(p_img_path).convert('RGB')).unsqueeze(0).to(device)
            
            # Extract features
            x1_feat = model.extract_features(x1_img).squeeze().cpu().numpy()
            x2_feat = model.extract_features(x2_img).squeeze().cpu().numpy()
            p_feat = model.extract_features(p_img).squeeze().cpu().numpy()
            
            # Calculate visual distances
            d1 = np.linalg.norm(x1_feat - p_feat)
            d2 = np.linalg.norm(x2_feat - p_feat)
            
            # Ground truth (evolutionary distance)
            ed1 = evo_distances_tensor[species_indices[x1], species_indices[p]].cpu().item()
            ed2 = evo_distances_tensor[species_indices[x2], species_indices[p]].cpu().item()
            true_closer = x1 if ed1 < ed2 else x2
            
            # Apply threshold for similarity
            sim1 = 1 if d1 <= threshold else 0
            sim2 = 1 if d2 <= threshold else 0
            
            # Prediction based on similarity
            if sim1 == 0 and sim2 == 0:
                # Both outside threshold, random guess
                prediction = np.random.choice([x1, x2])
            else:
                # Based on similarity ratio
                p_x1 = sim1 / (sim1 + sim2) if (sim1 + sim2) > 0 else 0.5
                prediction = x1 if np.random.random() < p_x1 else x2
            
            if prediction == true_closer:
                g_correct += 1
            g_total += 1
    
    return g_correct/g_total if g_total > 0 else 0

def get_random_image_path(species):
    """Helper to get a random image path for a species"""
    species_idx = class_names.index(species) + 1
    species_dir = os.path.join(images_dir, f"{species_idx:03d}.{species}")
    
    try:
        img_files = os.listdir(species_dir)
        if not img_files:
            return None
        return os.path.join(species_dir, np.random.choice(img_files))
    except:
        return None

# 6. Complete training and evaluation loop
def train_with_gi_tradeoff(alpha=0.5, epochs=10, batch_size=16, lr=0.001, random_state=42, 
                          gradient_accumulation_steps=1):
    """
    Train the model with controllable G-I tradeoff and evaluate both in-distribution and OOD
    """
    # Lower default batch size if using GPU to save memory
    if torch.cuda.is_available() and batch_size > 8:
        orig_batch_size = batch_size
        batch_size = min(batch_size, 8)  # Limit batch size on GPU
        gradient_accumulation_steps = max(gradient_accumulation_steps, orig_batch_size // batch_size)
        print(f"Adjusted batch size to {batch_size} with {gradient_accumulation_steps} gradient accumulation steps")
    
    # Create timestamp for folder names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories for models and results
    model_dir = f"../data/models/{timestamp}"
    results_dir = f"../results/models_{timestamp}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving models to: {model_dir}")
    print(f"Saving results to: {results_dir}")
    
    print(f"Training using: {device}")
    print_gpu_memory_usage("Before data prep")
    
    # Create data splits
    data_splits = create_data_splits(available_species, 
                                    test_size=0.2, 
                                    ood_size=0.15, 
                                    random_state=random_state)
    
    train_species = data_splits['train_species']
    ood_species = data_splits['ood_species']
    image_splits = data_splits['image_splits']
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Species indices for evolutionary distances
    species_indices = {sp: i for i, sp in enumerate(available_species)}
    
    # Free memory before creating datasets
    free_memory()
    print_gpu_memory_usage("Before dataset creation")
    
    # Create datasets
    train_dataset = BirdDataset(available_species, features_by_species, 
                               evo_distances, species_indices, 
                               image_splits, split='train', transform=transform)
    
    val_dataset = BirdDataset(available_species, features_by_species, 
                             evo_distances, species_indices, 
                             image_splits, split='val', transform=transform)
    
    test_dataset = BirdDataset(available_species, features_by_species, 
                              evo_distances, species_indices, 
                              image_splits, split='test', transform=transform)
    
    ood_dataset = BirdDataset(available_species, features_by_species, 
                             evo_distances, species_indices, 
                             image_splits, split='ood', transform=transform)
    
    # Free memory after dataset creation
    free_memory()
    print_gpu_memory_usage("After dataset creation")
    
    # Reduce num_workers on GPU to reduce memory usage
    num_workers = 2 if torch.cuda.is_available() else 4
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             num_workers=num_workers, pin_memory=True) 
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True)
    
    # Initialize model
    num_classes = len(train_species)
    
    # Free memory before model creation
    free_memory()
    print_gpu_memory_usage("Before model creation")
    
    model = GITradeoffModel(num_classes, alpha=alpha).to(device)
    
    # Loss functions
    id_criterion = nn.CrossEntropyLoss()
    
    # Move evo_distances to CPU to save GPU memory and only send to GPU when needed
    evo_distances_cpu = evo_distances
    if isinstance(evo_distances_cpu, torch.Tensor) and evo_distances_cpu.is_cuda:
        evo_distances_cpu = evo_distances_cpu.cpu()
    
    gen_criterion = EvolutionaryDistanceLoss(species_indices, evo_distances_cpu).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    
    print_gpu_memory_usage("Before training")
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    
    # In-distribution metrics (for set threshold of 15.0)
    id_scores = []      # I-score (identification)
    g_scores = []       # G-score (generalization)
    
    # Out-of-distribution metrics (for set threshold of 15.0)
    ood_g_scores = []   # OOD G-score
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Training loop
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            species = batch['species']
            
            # Free some memory before forward pass
            if batch_idx % 10 == 0:
                free_memory()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                features = outputs['features']
                logits = outputs['logits']
                
                id_loss = id_criterion(logits, labels)
                gen_loss = gen_criterion(features, species)
                
                loss = (1 - alpha) * id_loss + alpha * gen_loss
            
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
            
            # Explicitly free memory for large objects
            del images, labels, outputs, features, logits, loss, id_loss, gen_loss
            
            # Add periodic memory cleanup during training
            if (batch_idx + 1) % 10 == 0:
                free_memory()
        
        # Make sure to step optimizer for any remaining gradients at end of epoch
        if len(train_loader) % gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
        
        # Free memory before validation
        free_memory()
        print_gpu_memory_usage(f"Epoch {epoch+1} after training")
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                species = batch['species']
                
                outputs = model(images)
                features = outputs['features']
                logits = outputs['logits']
                
                id_loss = id_criterion(logits, labels)
                gen_loss = gen_criterion(features, species)
                
                loss = (1 - alpha) * id_loss + alpha * gen_loss
                val_loss += loss.item()
                
                # Free memory
                del images, labels, outputs, features, logits, loss, id_loss, gen_loss
        
        # Average losses
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Free memory before evaluation
        free_memory()
        print_gpu_memory_usage(f"Epoch {epoch+1} after validation")
        
        # Evaluate metrics with set threshold of 20.0
        i_score = evaluate_identification(model, test_loader, device, threshold=20.0)
        id_scores.append(i_score)
        
        # Free memory after each evaluation step
        free_memory()
        
        g_score = evaluate_generalization(model, threshold=20.0, 
                                          species_indices=species_indices,
                                          evo_distances=evo_distances_cpu,
                                          available_species=train_species,
                                          features_by_species=features_by_species,
                                          device=device, n_trials=500)
        g_scores.append(g_score)
        
        # Free memory after each evaluation step
        free_memory()
        
        ood_g_score = evaluate_generalization(model, threshold=20.0, 
                                             species_indices=species_indices,
                                             evo_distances=evo_distances_cpu,
                                             available_species=ood_species,
                                             features_by_species=features_by_species,
                                             device=device, n_trials=200)
        ood_g_scores.append(ood_g_score)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"I-score: {i_score:.4f} | G-score: {g_score:.4f} | OOD G-score: {ood_g_score:.4f}")
        
        # Add periodic GPU memory cleanup
        free_memory()
        print_gpu_memory_usage(f"Epoch {epoch+1} after evaluation")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model at EVERY epoch
        model_path = os.path.join(model_dir, f"model_alpha{alpha}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'i_score': i_score,
            'g_score': g_score,
            'ood_g_score': ood_g_score,
        }, model_path)
        print(f"Saved model for epoch {epoch+1}")
        
        # Track best model for final evaluation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print(f"New best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
            
        # Additional memory cleanup at end of each epoch
        free_memory()
    
    # Use the best model for final evaluation
    best_model_path = os.path.join(model_dir, f"model_alpha{alpha}_epoch{best_epoch+1}.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define thresholds to evaluate from 0 to 32
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # thresholds = np.linspace(0, 32, 33)
    
    # Evaluate final model across all thresholds
    threshold_results = {}
    
    for threshold in thresholds:
        print(f"Evaluating with threshold={threshold:.1f}")
        
        # In-distribution evaluation
        i_score = evaluate_identification(model, test_loader, device, threshold=threshold, n_trials=1000)
        free_memory()  # Free memory after evaluation
        
        g_score = evaluate_generalization(model, threshold=threshold, 
                                         species_indices=species_indices,
                                         evo_distances=evo_distances_cpu,
                                         available_species=train_species,
                                         features_by_species=features_by_species,
                                         device=device, n_trials=1000)
        free_memory()  # Free memory after evaluation
        
        # OOD evaluation
        ood_g_score = evaluate_generalization(model, threshold=threshold, 
                                             species_indices=species_indices,
                                             evo_distances=evo_distances_cpu,
                                             available_species=ood_species,
                                             features_by_species=features_by_species,
                                             device=device, n_trials=500)
        free_memory()  # Free memory after evaluation
        
        # Calculate theoretical scores
        g_theoretical = theoretical_G_score(threshold, evo_distances)
        i_theoretical = theoretical_I_score(threshold, evo_distances)
        
        threshold_results[threshold] = {
            'i_score': i_score,
            'g_score': g_score,
            'ood_g_score': ood_g_score,
            'i_theoretical': i_theoretical,
            'g_theoretical': g_theoretical
        }
        
        print(f"Threshold: {threshold:.1f} | I-score: {i_score:.4f} | G-score: {g_score:.4f} | OOD G-score: {ood_g_score:.4f}")
        print(f"Theoretical: I-score: {i_theoretical:.4f} | G-score: {g_theoretical:.4f}")
    
    # Final memory cleanup before plotting
    free_memory()
    
    # Plot G-I tradeoff curve for different thresholds
    plt.figure(figsize=(12, 10))
    
    # Extract scores for plotting
    thresh_values = list(threshold_results.keys())
    i_scores_by_thresh = [threshold_results[t]['i_score'] for t in thresh_values]
    g_scores_by_thresh = [threshold_results[t]['g_score'] for t in thresh_values]
    i_theo_by_thresh = [threshold_results[t]['i_theoretical'] for t in thresh_values]
    g_theo_by_thresh = [threshold_results[t]['g_theoretical'] for t in thresh_values]
    
    # Create color gradient based on threshold
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(thresh_values), max(thresh_values))
    colors = cmap(norm(thresh_values))
    
    # Plot empirical results
    for i in range(len(thresh_values)):
        if i > 0:
            plt.plot([g_scores_by_thresh[i-1], g_scores_by_thresh[i]], 
                     [i_scores_by_thresh[i-1], i_scores_by_thresh[i]], 
                     '-', color=colors[i], linewidth=2.5, alpha=0.7)
        plt.plot(g_scores_by_thresh[i], i_scores_by_thresh[i], 'o', 
                 color=colors[i], markersize=8, 
                 markeredgecolor='white', markeredgewidth=1)
    
    # Plot theoretical curve
    plt.plot(g_theo_by_thresh, i_theo_by_thresh, 'r--', linewidth=3, 
             label="Theoretical (Miller's Law)", alpha=0.8)
    
    plt.xlabel('Generalization (G)', fontsize=14, fontweight='bold')
    plt.ylabel('Identification (I)', fontsize=14, fontweight='bold')
    plt.title(f'G-I Tradeoff: Empirical vs Theoretical (α={alpha})', 
              fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Annotate some points
    annotation_indices = np.linspace(0, len(thresh_values)-1, 7, dtype=int)
    for i in annotation_indices:
        plt.annotate(f"ε={thresh_values[i]:.1f}", 
                   (g_scores_by_thresh[i], i_scores_by_thresh[i]),
                   xytext=(10, 0), textcoords="offset points",
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax = plt.gca()  # Get current axes reference
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Threshold (ε)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'gi_threshold_tradeoff_alpha{alpha}_with_theory.png'), dpi=300)
    plt.close()
    
    # Save threshold results to a CSV file
    with open(os.path.join(results_dir, f'threshold_results_alpha{alpha}.csv'), 'w') as f:
        f.write('threshold,i_score,g_score,ood_g_score,i_theoretical,g_theoretical\n')
        for threshold in threshold_results:
            result = threshold_results[threshold]
            f.write(f"{threshold},{result['i_score']},{result['g_score']},{result['ood_g_score']},{result['i_theoretical']},{result['g_theoretical']}\n")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'i_scores': id_scores,
        'g_scores': g_scores,
        'ood_g_scores': ood_g_scores,
        'threshold_results': threshold_results,
        'timestamp': timestamp  # Return timestamp for use in run_experiments
    }

# Modified run_experiments function to handle memory better
def run_experiments(alpha_values=None, random_seeds=None):
    # Create single timestamp for all experiments
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results/models_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze the space properties first
    print("\n=== Analyzing Space Properties ===\n")
    is_disc = is_discriminative(evo_distances)
    print(f"Is the evolutionary distance space discriminative? {is_disc}")
    
    alpha_term = calculate_alpha_term(evo_distances)
    print(f"Alpha term (equidistant sphere measure): {alpha_term}")
    
    # Calculate the theoretical max G-score
    if is_disc:
        g_max = 3/4
        print(f"Space is discriminative, theoretical Gmax = {g_max}")
    else:
        # For non-discriminative spaces, calculate the max from Theorem 1
        g_max = (1/2) * (1 + alpha_term) + 0.5 - 0.5**2
        print(f"Space is not discriminative, calculated Gmax = {g_max:.4f}")
    
    # Use provided alpha values or default list
    if alpha_values is None:
        # Default alpha values to try - reduce number for memory savings
        alphas = [0.0, 0.5, 1.0]  # Simplified set for memory constraints
    else:
        alphas = alpha_values
    
    # Use provided random seeds or default to a single seed
    if random_seeds is None:
        random_seeds = [42]
    
    print(f"Running experiments with alpha values: {alphas}")
    print(f"Using {len(random_seeds)} random seeds: {random_seeds}")
    
    # Store results for each alpha and seed combination
    results = {}
    
    for alpha in alphas:
        results[alpha] = []
        
        for seed in random_seeds:
            print(f"\n=== Training with alpha={alpha}, seed={seed} ===\n")
            # Explicitly free memory before each new experiment
            free_memory()
            print_gpu_memory_usage(f"Before starting alpha={alpha}, seed={seed}")
            
            result = train_with_gi_tradeoff(alpha=alpha, epochs=15, random_state=seed, 
                                            batch_size=8, gradient_accumulation_steps=4)
            results[alpha].append(result)
            
            # Free memory after each complete experiment
            free_memory()
            print_gpu_memory_usage(f"After completing alpha={alpha}, seed={seed}")
    
    # Skip combined plots if only running a single alpha value
    if len(alphas) > 1:
        # Free memory before plotting
        free_memory()
        print_gpu_memory_usage("Before creating final plots")
        
        # Plot all trajectories together with theoretical curve
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
        
        # First plot the theoretical curve
        thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        g_theo = []
        i_theo = []
        
        for threshold in thresholds:
            g_theo.append(theoretical_G_score(threshold, evo_distances))
            i_theo.append(theoretical_I_score(threshold, evo_distances))
        
        plt.plot(g_theo, i_theo, 'r--', linewidth=3, 
                 label="Theoretical (Miller's Law)", alpha=0.8)
        
        # Then plot empirical trajectories for each alpha and seed
        for i, alpha in enumerate(alphas):
            for j, seed_results in enumerate(results[alpha]):
                g_scores = seed_results['g_scores']
                i_scores = seed_results['i_scores']
                
                # Use different line styles for different seeds
                linestyle = ['-', '--', '-.', ':', '-'][j]
                
                plt.plot(g_scores, i_scores, 'o'+linestyle, color=colors[i], 
                         linewidth=1.5, alpha=0.7, label=f'α={alpha}, seed={random_seeds[j]}' if j == 0 else "")
                
                plt.scatter(g_scores[0], i_scores[0], s=50, 
                            facecolors='none', edgecolors=colors[i], alpha=0.7)
                plt.scatter(g_scores[-1], i_scores[-1], s=50, 
                            color=colors[i], alpha=0.7)
        
        plt.xlabel('Generalization (G)', fontsize=14, fontweight='bold')
        plt.ylabel('Identification (I)', fontsize=14, fontweight='bold')
        plt.title('G-I Training Trajectories with Different α Values and Seeds', 
                  fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'gi_all_trajectories_with_theory.png'), dpi=300)
        plt.close()
        
        # Plot OOD performance
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for i, alpha in enumerate(alphas):
            for j, seed_results in enumerate(results[alpha]):
                # Use different line styles for different seeds
                linestyle = ['-', '--', '-.', ':', '-'][j]
                plt.plot(seed_results['g_scores'], linestyle, color=colors[i], alpha=0.7, 
                         label=f'α={alpha}, seed={random_seeds[j]}' if j == 0 else "")
        plt.xlabel('Epoch')
        plt.ylabel('In-distribution G-score')
        plt.title('In-distribution Generalization')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, alpha in enumerate(alphas):
            for j, seed_results in enumerate(results[alpha]):
                # Use different line styles for different seeds
                linestyle = ['-', '--', '-.', ':', '-'][j]
                plt.plot(seed_results['ood_g_scores'], linestyle, color=colors[i], alpha=0.7,
                         label=f'α={alpha}, seed={random_seeds[j]}' if j == 0 else "")
        plt.xlabel('Epoch')
        plt.ylabel('OOD G-score')
        plt.title('Out-of-distribution Generalization')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ood_performance.png'), dpi=300)
        plt.close()
    
    # Final memory cleanup
    free_memory()
    print_gpu_memory_usage("End of experiments")
    
    return results

# Modify main execution block
if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run bird species generalization experiments')
    parser.add_argument('--alpha', type=float, nargs='+', help='Alpha value(s) to train with (e.g., 0.5 0.75)')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--num-seeds', type=int, default=5, help='Number of random seeds to use (default: 5)')
    parser.add_argument('--seeds', type=int, nargs='+', help='Specific random seeds to use (overrides num-seeds)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # For reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Update training function parameters
    train_with_gi_tradeoff.__defaults__ = (0.5, args.epochs, args.batch_size, 0.001, 42, args.grad_accum)
    
    # If specific seeds are provided, use those
    if args.seeds:
        random_seeds = args.seeds
    else:
        # Otherwise generate the specified number of seeds
        random_seeds = list(range(42, 42 + args.num_seeds))
    
    # Override the seeds in run_experiments
    run_experiments.__defaults__ = (None,)
    
    # Log the experiment configuration
    print(f"Starting experiment with:")
    print(f"  Alpha values: {args.alpha}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.grad_accum}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random seeds: {random_seeds}")
    
    # Monkey patch the run_experiments function to use our seeds
    original_run_experiments = run_experiments
    def patched_run_experiments(alpha_values=None):
        # Use seeds from outer scope
        # Create a temporary version of run_experiments that uses our seeds
        old_body = original_run_experiments.__code__
        from types import FunctionType
        new_func = FunctionType(old_body, original_run_experiments.__globals__, 
                              original_run_experiments.__name__, 
                              original_run_experiments.__defaults__,
                              original_run_experiments.__closure__)
        # Call with our random seeds
        return new_func(alpha_values, random_seeds)
    
    # Replace run_experiments with our patched version
    run_experiments = patched_run_experiments
    
    try:
        # Ensure clean memory state at start
        free_memory()
        print_gpu_memory_usage("Before starting experiments")
        
        # Run with specified alpha values if provided
        if args.alpha:
            results = run_experiments(alpha_values=args.alpha)
        else:
            # Default behavior - run all alpha values
            results = run_experiments()
            
        # Clean up GPU memory after we're done
        free_memory()
        print_gpu_memory_usage("Final state")
    except Exception as e:
        print(f"Error during execution: {e}")
        # Make sure to clean up GPU memory even if there's an error
        free_memory()
        print_gpu_memory_usage("Error state")