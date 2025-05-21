#!/usr/bin/env python3
"""
Evolutionary generalization of bird species with maximum diversity data split.

This version selects the 15 most evolutionary distant species for training
and tests generalization on the remaining species.
"""

import dendropy
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import argparse
import datetime
import shutil

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

# Import from original script
# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.birdsbirdsbirds_v2 import (
    # Model and loss functions
    GITradeoffModel, 
    EvolutionaryDistanceLoss,
    # Evaluation functions
    evaluate_identification,
    evaluate_generalization,
    run_tests,
    theoretical_G_score,
    theoretical_I_score,
    # Utility functions
    calculate_average_ball_measure,
    calculate_alpha_term,
    log_step,
    BirdDataset,  # We'll override this with our modified version
)

# Define global device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Add diagnostic function to track progress 
def log_step(message):
    """Print timestamped log message"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

log_step("Starting script execution")

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

def select_diverse_species(evo_distances, available_species, num_train_species=15, seed=42):
    """
    Select a set of species with maximum evolutionary diversity for training.
    
    Args:
        evo_distances: Matrix of evolutionary distances between species
        available_species: List of all available species
        num_train_species: Number of species to select for training (default: 15)
        seed: Random seed for reproducibility
        
    Returns:
        train_species: List of selected diverse species for training
        remaining_species: List of remaining species for testing
    """
    np.random.seed(seed)
    
    # Create indices mapping for quick lookup
    species_indices = {sp: i for i, sp in enumerate(available_species)}
    
    # First, find the pair of species with the maximum distance
    max_dist = 0
    max_pair = None
    
    for i, sp1 in enumerate(available_species):
        for j, sp2 in enumerate(available_species[i+1:], i+1):
            dist = evo_distances[i, j]
            if dist > max_dist:
                max_dist = dist
                max_pair = (sp1, sp2)
    
    if max_pair is None:
        raise ValueError("Could not find species pair with maximum distance")
    
    print(f"Starting with species pair with maximum distance: {max_pair[0]}, {max_pair[1]} (dist: {max_dist:.4f})")
    
    # Initialize selected species with the max pair
    selected_species = list(max_pair)
    
    # Greedy algorithm: iteratively add the species with the maximum
    # average distance to already selected species
    while len(selected_species) < num_train_species and len(selected_species) < len(available_species):
        max_avg_dist = 0
        next_species = None
        
        for sp in available_species:
            if sp in selected_species:
                continue
                
            # Calculate average distance to selected species
            total_dist = sum(evo_distances[species_indices[sp], species_indices[sel_sp]] 
                             for sel_sp in selected_species)
            avg_dist = total_dist / len(selected_species)
            
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                next_species = sp
        
        if next_species is None:
            break
            
        selected_species.append(next_species)
        print(f"Added species {next_species} with avg distance {max_avg_dist:.4f}")
    
    # Remaining species for testing
    remaining_species = [sp for sp in available_species if sp not in selected_species]
    
    print(f"Selected {len(selected_species)} diverse species for training")
    print(f"Remaining {len(remaining_species)} species for testing")
    
    return selected_species, remaining_species

def create_diverse_data_splits(available_species, features_by_species, evo_distances, 
                               num_train_species=15, test_size=0.2, random_state=42):
    """
    Create data splits using the most evolutionary diverse species for training.
    
    Args:
        available_species: List of all available species
        features_by_species: Dictionary of features for each species
        evo_distances: Matrix of evolutionary distances
        num_train_species: Number of diverse species to select for training
        test_size: Proportion of in-distribution test images
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with all split information
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Filter to species that have features extracted
    species_with_features = [sp for sp in available_species if sp in features_by_species]
    
    # Select diverse species for training
    train_species, test_species = select_diverse_species(
        evo_distances, species_with_features, num_train_species, random_state)
    
    print(f"Using {len(train_species)} diverse species for training")
    print(f"Using {len(test_species)} remaining species for testing")
    
    # Now, for training species, split images into train/val/test
    image_splits = {}
    
    # Image directory from the CUB dataset
    cub_root = '../CUB_200_2011'
    images_dir = os.path.join(cub_root, 'images')
    
    # Read class names if not already available
    if 'class_names' not in globals():
        with open(os.path.join(cub_root, 'classes.txt'), 'r') as f:
            class_names = [line.split('.')[1].strip() for line in f.readlines()]
    else:
        class_names = globals()['class_names']
    
    for species in available_species:
        if species not in features_by_species:
            continue
            
        # Get all image paths for this species
        species_idx = class_names.index(species) + 1
        species_dir = os.path.join(images_dir, f"{species_idx:03d}.{species}")
        image_files = os.listdir(species_dir)
        
        if species in train_species:
            # For training species, split into train/val/test
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
            # Test species - use all images for generalization testing
            # We'll put a small portion in validation for model selection
            val_imgs, test_imgs = train_test_split(
                image_files, test_size=0.8, random_state=random_state)
            
            image_splits[species] = {
                'train': [],  # No training images
                'val': val_imgs,  # Small set for validation
                'test': test_imgs,  # Most for testing
                'split_type': 'generalization'
            }
    
    return {
        'train_species': train_species,
        'test_species': test_species,
        'image_splits': image_splits
    }

def main():
    """Main function to run the experiment with the diverse data split approach"""
    parser = argparse.ArgumentParser(description='Run bird species generalization with diverse training species')
    parser.add_argument('--num-species', type=int, default=15, help='Number of diverse species to use for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for balancing ID vs generalization loss')
    args = parser.parse_args()
    
    # Create directories for models and results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"../data/new_datasplit/models_{timestamp}"
    results_dir = f"../results/new_datasplit/models_{timestamp}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving models to: {model_dir}")
    print(f"Saving results to: {results_dir}")
    
    # Copy this script to the model directory for reproducibility
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(model_dir, "script.py"))
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # For reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 1. Load the Newick tree and process it
    log_step("Loading Newick tree file...")
    tree = dendropy.Tree.get(path="../data/birds_species.nwk", schema="newick")
    log_step("Tree file loaded successfully")
    
    # 2. Load CUB dataset information
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
    
    # 3. Map common names to scientific names (use imported code or copy from original)
    # This is where we need to match CUB dataset common names to scientific names in the tree
    log_step("Mapping common names to scientific names...")
    
    # We need to import both common_to_scientific mapping and features_by_species
    # from the original script since the mapping is complex
    
    # First, try to match the common names to scientific names (simplified from original)
    common_to_scientific = {}
    
    # We'll try to match each common name word with scientific name words
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
            
            # Check for match
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
    
    # Add some specific, verified mappings for common birds
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
    
    # 4. Build the evolutionary distance matrix
    log_step("Building evolutionary distance matrix...")
    
    # Get species that we can use (have both common and scientific names)
    available_species = []
    for species in class_names:
        if species in common_to_scientific:
            scientific_name = common_to_scientific[species]
            if tree.find_node_with_taxon_label(scientific_name) is not None:
                available_species.append(species)
    
    print(f"Found {len(available_species)} species in both datasets that can be used")
    
    # Calculate evolutionary distances between all available species
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
    
    # 5. Load or extract features for each species
    log_step("Loading ResNet50 model for feature extraction...")
    
    # Load ResNet50 model
    try:
        # First try loading the model with pretrained weights from local cache (if it exists)
        weights_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
        if os.path.exists(weights_path):
            log_step(f"Found local weights at {weights_path}")
            # Load model without downloading, will use local weights
            model_resnet = models.resnet50()
            model_resnet.load_state_dict(torch.load(weights_path))
            
            log_step("Loaded ResNet50 with pretrained weights from local cache")
        else:
            # Initialize with random weights - no internet download
            log_step("No pretrained weights found locally, initializing with random weights")
            model_resnet = models.resnet50(pretrained=False)
            log_step("Initialized ResNet50 with random weights")
    except Exception as e:
        log_step(f"Error during model loading: {e}")
        log_step("Falling back to random initialization")
        model_resnet = models.resnet50(pretrained=False)
    
    log_step("ResNet50 model loaded successfully")
    
    # Create feature extractor
    feature_extractor = torch.nn.Sequential(*list(model_resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    log_step("Feature extractor ready for inference")
    
    # Set up transforms for feature extraction
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Extract features for each species or load pre-extracted features if available
    features_cache_path = "../data/features_cache.npy"
    if os.path.exists(features_cache_path):
        # Load cached features
        log_step(f"Loading cached features from {features_cache_path}")
        features_cache = np.load(features_cache_path, allow_pickle=True).item()
        features_by_species = features_cache
        log_step(f"Loaded features for {len(features_by_species)} species")
    else:
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
        
        # Save features cache
        log_step(f"Saving features to cache: {features_cache_path}")
        os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
        np.save(features_cache_path, features_by_species)
        log_step("Features cache saved successfully")
    
    # 6. Create our new diverse data splits
    log_step("Creating diverse data splits...")
    data_splits = create_diverse_data_splits(
        available_species, 
        features_by_species, 
        evo_distances,
        num_train_species=args.num_species
    )
    
    train_species = data_splits['train_species']
    test_species = data_splits['test_species'] 
    image_splits = data_splits['image_splits']
    
    # 7. Set up datasets and dataloaders with our special split
    log_step("Setting up datasets and dataloaders...")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Species indices for evolutionary distances
    species_indices = {sp: i for i, sp in enumerate(available_species)}
    
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=4, pin_memory=True)
    
    # 8. Train the model with diverse species
    log_step("Setting up model, optimizer, and loss functions...")
    
    num_classes = len(train_species)
    model = GITradeoffModel(num_classes, alpha=args.alpha).to(device)
    
    # Loss functions
    id_criterion = nn.CrossEntropyLoss()
    gen_criterion = EvolutionaryDistanceLoss(species_indices, evo_distances).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    
    # Train the model
    log_step("Starting training...")
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    id_scores = []
    g_scores = []
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Training loop
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            species = batch['species']
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                features = outputs['features']
                logits = outputs['logits']
                
                id_loss = id_criterion(logits, labels)
                gen_loss = gen_criterion(features, species)
                
                loss = (1 - args.alpha) * id_loss + args.alpha * gen_loss
            
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum  # Scale back for reporting
        
        # Make sure to step optimizer for any remaining gradients at end of epoch
        if len(train_loader) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
        
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
                
                loss = (1 - args.alpha) * id_loss + args.alpha * gen_loss
                val_loss += loss.item()
        
        # Average losses
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Evaluate metrics with set threshold of 15.0
        # Separate evaluation for in-distribution and generalization species
        
        # In-distribution evaluation (train species)
        threshold = 15.0  # Default threshold for evaluation during training
        i_score = evaluate_identification(model, test_loader, device, threshold=threshold)
        id_scores.append(i_score)
        
        # Generalization evaluation (test species)
        g_score = evaluate_generalization(model, threshold=threshold, 
                                         species_indices=species_indices,
                                         evo_distances=evo_distances,
                                         available_species=train_species,
                                         features_by_species=features_by_species,
                                         device=device, n_trials=500)
        g_scores.append(g_score)
        
        # Test generalization on the species not seen during training
        gen_g_score = evaluate_generalization(model, threshold=threshold, 
                                             species_indices=species_indices,
                                             evo_distances=evo_distances,
                                             available_species=test_species,
                                             features_by_species=features_by_species,
                                             device=device, n_trials=200)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"I-score: {i_score:.4f} | Train G-score: {g_score:.4f} | Test G-score: {gen_g_score:.4f}")
        
        # Add periodic GPU memory cleanup
        if torch.cuda.is_available() and (epoch+1) % 3 == 0:  # Every 3 epochs
            torch.cuda.empty_cache()
            print("Cleaned GPU cache")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model at every epoch
        model_path = os.path.join(model_dir, f"model_diverse_alpha{args.alpha}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'i_score': i_score,
            'g_score': g_score,
            'gen_g_score': gen_g_score,
        }, model_path)
        print(f"Saved model for epoch {epoch+1}")
        
        # Track best model for final evaluation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print(f"New best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
    
    # Use the best model for final evaluation
    best_model_path = os.path.join(model_dir, f"model_diverse_alpha{args.alpha}_epoch{best_epoch+1}.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define thresholds to evaluate G-I tradeoff curve
    thresholds = np.linspace(0, 50, 33)
    
    # Evaluate final model across all thresholds
    threshold_results = {}
    
    for threshold in thresholds:
        print(f"Evaluating with threshold={threshold:.1f}")
        
        # In-distribution evaluation
        i_score = evaluate_identification(model, test_loader, device, threshold=threshold, n_trials=1000)
        g_score = evaluate_generalization(model, threshold=threshold, 
                                         species_indices=species_indices,
                                         evo_distances=evo_distances,
                                         available_species=train_species,
                                         features_by_species=features_by_species,
                                         device=device, n_trials=1000)
        
        # Generalization evaluation - species not seen during training
        gen_g_score = evaluate_generalization(model, threshold=threshold, 
                                             species_indices=species_indices,
                                             evo_distances=evo_distances,
                                             available_species=test_species,
                                             features_by_species=features_by_species,
                                             device=device, n_trials=500)
        
        # Calculate theoretical scores
        g_theoretical = theoretical_G_score(threshold, evo_distances)
        i_theoretical = theoretical_I_score(threshold, evo_distances)
        
        threshold_results[threshold] = {
            'i_score': i_score,
            'g_score': g_score,
            'gen_g_score': gen_g_score,
            'i_theoretical': i_theoretical,
            'g_theoretical': g_theoretical
        }
        
        print(f"Threshold: {threshold:.1f} | I-score: {i_score:.4f} | Train G-score: {g_score:.4f} | Test G-score: {gen_g_score:.4f}")
        print(f"Theoretical: I-score: {i_theoretical:.4f} | G-score: {g_theoretical:.4f}")
    
    # Plot G-I tradeoff curves
    plt.figure(figsize=(12, 10))
    
    # Extract scores for plotting
    thresh_values = list(threshold_results.keys())
    i_scores_by_thresh = [threshold_results[t]['i_score'] for t in thresh_values]
    g_scores_by_thresh = [threshold_results[t]['g_score'] for t in thresh_values]
    gen_g_scores_by_thresh = [threshold_results[t]['gen_g_score'] for t in thresh_values]
    i_theo_by_thresh = [threshold_results[t]['i_theoretical'] for t in thresh_values]
    g_theo_by_thresh = [threshold_results[t]['g_theoretical'] for t in thresh_values]
    
    # Create color gradient based on threshold
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(thresh_values), max(thresh_values))
    colors = cmap(norm(thresh_values))
    
    # Plot empirical results for train species
    plt.plot(g_scores_by_thresh, i_scores_by_thresh, '-', color='blue', 
             linewidth=2.5, label='In-distribution', alpha=0.8)
    
    # Plot empirical results for test species (generalization)
    plt.plot(gen_g_scores_by_thresh, i_scores_by_thresh, '-', color='green', 
             linewidth=2.5, label='Generalization (diverse test)', alpha=0.8)
    
    # Plot theoretical curve
    plt.plot(g_theo_by_thresh, i_theo_by_thresh, 'r--', linewidth=3, 
             label="Theoretical (Miller's Law)", alpha=0.8)
    
    # Plot individual points with color gradient
    for i in range(len(thresh_values)):
        plt.plot(g_scores_by_thresh[i], i_scores_by_thresh[i], 'o', color=colors[i], 
                 markersize=6, markeredgecolor='white', markeredgewidth=1)
        plt.plot(gen_g_scores_by_thresh[i], i_scores_by_thresh[i], 's', color=colors[i], 
                 markersize=6, markeredgecolor='white', markeredgewidth=1)
    
    plt.xlabel('Generalization (G)', fontsize=14, fontweight='bold')
    plt.ylabel('Identification (I)', fontsize=14, fontweight='bold')
    plt.title('G-I Tradeoff with Diverse Species Training (10-20 most distinct species)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Annotate some points
    annotation_indices = np.linspace(0, len(thresh_values)-1, 6, dtype=int)
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
    plt.savefig(os.path.join(results_dir, 'gi_diverse_species_tradeoff.png'), dpi=300)
    plt.close()
    
    # Save threshold results to a CSV file
    with open(os.path.join(results_dir, f'threshold_results_diverse_alpha{args.alpha}.csv'), 'w') as f:
        f.write('threshold,i_score,g_score,gen_g_score,i_theoretical,g_theoretical\n')
        for threshold in threshold_results:
            result = threshold_results[threshold]
            f.write(f"{threshold},{result['i_score']},{result['g_score']},{result['gen_g_score']},{result['i_theoretical']},{result['g_theoretical']}\n")
    
    log_step("Experiment complete!")
    return {
        'model_dir': model_dir,
        'results_dir': results_dir,
        'train_species': train_species,
        'test_species': test_species
    }

if __name__ == "__main__":
    main() 