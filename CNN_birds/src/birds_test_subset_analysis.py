#!/usr/bin/env python3
"""
Analyze the subset of birds used in model testing
Focuses on the evolutionary distance matrix from birdsbirdsbirds_v2.py
"""
import dendropy
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.spatial.distance import squareform
import datetime

# Try to import persistent homology libraries
try:
    import ripser
    USE_RIPSER = True
    print("Using Ripser for persistent homology")
except ImportError:
    try:
        import gudhi
        USE_RIPSER = False
        print("Using GUDHI for persistent homology")
    except ImportError:
        print("WARNING: No persistent homology library found. Install ripser or gudhi.")
        USE_RIPSER = None

def log_step(message):
    """Print timestamped log message"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_bird_data(test_split=0.2, ood_split=0.15, random_state=42):
    """
    Load the bird data and create train/test splits identical to birdsbirdsbirds_v2.py
    Reproduces the exact setup used in the model
    """
    log_step("Loading bird data...")
    
    # Check if we have cached data
    cache_file = "../data/test_subset_data_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            log_step("Loaded data from cache")
            return data
    
    # Load tree
    log_step("Loading Newick tree file...")
    tree = dendropy.Tree.get(path="../data/birds_species.nwk", schema="newick")
    log_step("Tree file loaded successfully")
    
    # Load CUB dataset
    log_step("Loading CUB dataset names...")
    cub_root = '../CUB_200_2011'
    
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
    
    # Add manual mappings (copied from birdsbirdsbirds_v2.py)
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
    
    # First add manual mappings
    common_to_scientific = {}
    for common, sci in manual_mappings.items():
        if common in class_names:
            # Check if the scientific name is in the tree
            found = False
            for taxon in tree.taxon_namespace:
                if taxon.label == sci:
                    found = True
                    break
            
            if found:
                common_to_scientific[common] = sci
    
    # Add automated mappings for remaining species
    for common_name in class_names:
        if common_name in common_to_scientific:
            continue  # Already mapped
        
        # Clean common name (replace underscores with spaces)
        common_name_clean = common_name.replace('_', ' ')
        
        # Split the common name into words
        common_words = common_name_clean.lower().split()
        
        # Try to find matching scientific names
        potential_matches = []
        
        for sci_name in scientific_names:
            # Split scientific name - normally has genus and species
            sci_parts = sci_name.lower().split()
            
            # Check for match - simple scoring approach
            match_score = 0
            
            # Check if genus/species words appear in common name
            for sci_part in sci_parts:
                if len(sci_part) >= 5:  # Only use meaningful parts
                    for word in common_words:
                        if len(word) >= 5:  # Avoid short words
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
    
    # Calculate evolutionary distance matrix
    log_step("Calculating evolutionary distance matrix...")
    species_count = len(available_species)
    evo_distances = np.zeros((species_count, species_count))
    
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
        
        # Find leaf nodes
        for node in tree.leaf_node_iter():
            if node.taxon == taxon1:
                leaf1 = node
            if node.taxon == taxon2:
                leaf2 = node
        
        if leaf1 is None or leaf2 is None:
            return None
        
        # Calculate path-length distance
        try:
            mrca = tree.mrca(taxa=[taxon1, taxon2])
            if mrca is None:
                return 0.0
            
            # Calculate distances from leaves to MRCA
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
            
            return dist1 + dist2
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0.0
    
    for i, sp1 in enumerate(available_species):
        for j, sp2 in enumerate(available_species):
            if i == j:
                continue
            sci1 = common_to_scientific[sp1]
            sci2 = common_to_scientific[sp2]
            distance = get_evolutionary_distance(tree, sci1, sci2)
            if distance is not None:
                evo_distances[i, j] = distance
    
    log_step("Evolutionary distance matrix calculation complete")
    
    # Create train/test/OOD split exactly as in original script
    log_step("Creating train/test/OOD splits...")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # First, split species into those used for training and those held out (OOD)
    n_ood_species = max(1, int(len(available_species) * ood_split))
    train_species_idx = sorted(np.random.choice(
        range(len(available_species)), 
        len(available_species) - n_ood_species, 
        replace=False
    ))
    
    train_species = [available_species[i] for i in train_species_idx]
    ood_species = [sp for sp in available_species if sp not in train_species]
    
    print(f"Training species: {len(train_species)}")
    print(f"Out-of-distribution species: {len(ood_species)}")
    
    # Create train, in-distribution test, and OOD test evolutionary distance matrices
    train_indices = [i for i, sp in enumerate(available_species) if sp in train_species]
    ood_indices = [i for i, sp in enumerate(available_species) if sp in ood_species]
    
    train_evo_distances = evo_distances[np.ix_(train_indices, train_indices)]
    ood_evo_distances = evo_distances[np.ix_(ood_indices, ood_indices)]
    
    # Store data
    data = {
        'tree': tree,
        'class_names': class_names,
        'common_to_scientific': common_to_scientific,
        'available_species': available_species,
        'evo_distances': evo_distances,
        'train_species': train_species,
        'train_indices': train_indices,
        'train_evo_distances': train_evo_distances,
        'ood_species': ood_species,
        'ood_indices': ood_indices,
        'ood_evo_distances': ood_evo_distances
    }
    
    # Cache the data for future use
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    log_step("Data cached for future use")
    
    return data

def run_persistent_homology(distance_matrix, max_dim=1):
    """Compute persistent homology on a distance matrix"""
    if USE_RIPSER is None:
        print("No persistent homology library available")
        return None
    
    log_step(f"Computing persistent homology (dim 0 to {max_dim})...")
    
    # Ensure distance matrix is symmetric and non-negative
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    distance_matrix = np.maximum(0, distance_matrix)
    
    if USE_RIPSER:
        # Using Ripser.py
        result = ripser.ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)
        
        # Return the persistent homology diagrams
        return {
            'dgms': result['dgms'],  # List of diagrams, one for each dimension
            'cocycles': result.get('cocycles', None)
        }
    else:
        # Using GUDHI
        # Create Rips complex
        rips = gudhi.RipsComplex(distance_matrix=distance_matrix)
        
        # Create simplex tree with filtration
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dim+1)
        
        # Compute persistent homology
        persistence = simplex_tree.persistence()
        
        # Extract diagrams
        dgms = [[] for _ in range(max_dim+1)]
        for pair in persistence:
            dim, (birth, death) = pair[0], pair[1]
            if dim <= max_dim:
                dgms[dim].append([birth, death])
        
        return {
            'dgms': [np.array(dgm) for dgm in dgms]
        }

def plot_connected_components(diagrams, distance_matrix, species_names, 
                             highlight_epsilon=None, output_dir=None, suffix=''):
    """
    Plot connected components vs epsilon with option to highlight specific thresholds
    
    Args:
        diagrams: List of persistence diagrams, one for each dimension
        distance_matrix: Distance matrix used for PHom
        species_names: List of species names corresponding to distance_matrix
        highlight_epsilon: List of specific epsilon values to highlight
        output_dir: Directory to save output plots
        suffix: Suffix to add to output filename
    """
    # Use default output path if none specified
    if output_dir is None:
        output_dir = "."
    
    # Find the epsilon values where components merge (H0 bars die)
    if len(diagrams) > 0 and len(diagrams[0]) > 0:
        h0_diagram = diagrams[0]
        
        # Get finite death times (ignore inf)
        finite = ~np.isinf(h0_diagram[:, 1])
        death_times = h0_diagram[finite, 1]
        
        # Sort death times
        sorted_deaths = np.sort(death_times)
        
        # Get number of connected components at each threshold
        thresholds = np.linspace(0, sorted_deaths[-1]*1.2, 100)
        components = []
        
        for eps in thresholds:
            # Create adjacency matrix at this threshold
            adj_matrix = distance_matrix <= eps
            np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
            
            # Count connected components
            # Simple DFS implementation
            n = len(distance_matrix)
            visited = [False] * n
            count = 0
            
            def dfs(node):
                visited[node] = True
                for neighbor in range(n):
                    if adj_matrix[node, neighbor] and not visited[neighbor]:
                        dfs(neighbor)
            
            for i in range(n):
                if not visited[i]:
                    count += 1
                    dfs(i)
            
            components.append(count)
        
        # Plot the number of connected components vs epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, components, 'b-', linewidth=2.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epsilon (ε)', fontsize=14)
        plt.ylabel('Number of Connected Components', fontsize=14)
        plt.title('Connected Components vs. Epsilon', fontsize=16)
        
        # Add points at key transition thresholds
        transitions = []
        for i in range(1, len(components)):
            if components[i] < components[i-1]:
                transitions.append((thresholds[i], components[i]))
        
        # Plot the first few transitions
        num_to_show = min(5, len(transitions))
        for i in range(num_to_show):
            eps, comp = transitions[i]
            plt.plot(eps, comp, 'ro', markersize=8, alpha=0.7)
            plt.annotate(f"ε={eps:.2f}, {comp} components",
                        xy=(eps, comp),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Highlight specific epsilon values if requested
        if highlight_epsilon is not None:
            # For each threshold, find the number of components
            for eps in highlight_epsilon:
                # Find the closest threshold
                idx = np.argmin(np.abs(thresholds - eps))
                comp = components[idx]
                plt.axvline(x=eps, color='green', linestyle='--', alpha=0.5)
                plt.plot(eps, comp, 'go', markersize=10)
                plt.annotate(f"ε={eps:.2f}\n{comp} components",
                            xy=(eps, comp),
                            xytext=(0, -30),
                            textcoords="offset points",
                            fontsize=10,
                            arrowprops=dict(arrowstyle="->", color='green'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'connected_components_modelsubset{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved connected components plot to {save_path}")
        plt.close()
        
        # Return the components and thresholds for further analysis
        return thresholds, components

def plot_threshold_distribution(diagrams, distance_matrix, output_dir=None, suffix=''):
    """
    Plot histogram of critical merge thresholds (death values for H0)
    
    Args:
        diagrams: List of persistence diagrams, one for each dimension
        distance_matrix: Distance matrix used for PHom
        output_dir: Directory to save output plots
        suffix: Suffix to add to output filename
    """
    # Use default output path if none specified
    if output_dir is None:
        output_dir = "."
    
    # Find the epsilon values where components merge (H0 bars die)
    if len(diagrams) > 0 and len(diagrams[0]) > 0:
        h0_diagram = diagrams[0]
        
        # Get finite death times (ignore inf)
        finite = ~np.isinf(h0_diagram[:, 1])
        death_times = h0_diagram[finite, 1]
        
        # Create histogram of death times
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(death_times, bins=20, color='blue', alpha=0.7)
        
        # Add vertical lines for key thresholds (e.g., 10)
        key_thresholds = [10, 15, 20]
        for threshold in key_thresholds:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
            plt.text(threshold, max(counts)*0.9, f"ε={threshold}", 
                     rotation=90, verticalalignment='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        plt.xlabel('Epsilon (ε) at Component Merge', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Critical Merge Thresholds', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'threshold_distribution_modelsubset{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved threshold distribution plot to {save_path}")
        plt.close()

def analyze_gi_thresholds(train_thresholds, train_components, 
                          ood_thresholds=None, ood_components=None,
                          output_dir=None):
    """
    Analyze thresholds relevant to G-I tradeoff
    
    Args:
        train_thresholds: Array of threshold values for training species
        train_components: Array of component counts for training species
        ood_thresholds: Array of threshold values for OOD species
        ood_components: Array of component counts for OOD species
        output_dir: Directory to save output plots
    """
    # Use default output path if none specified
    if output_dir is None:
        output_dir = "."
    
    # Convert lists to numpy arrays if needed
    train_thresholds = np.array(train_thresholds)
    train_components = np.array(train_components)
    if ood_thresholds is not None:
        ood_thresholds = np.array(ood_thresholds)
    if ood_components is not None:
        ood_components = np.array(ood_components)
    
    # Calculate key points: 
    # 1. Where we lose 50% of components
    # 2. Where the space becomes fully connected
    n_initial_components = train_components[0]
    half_components = n_initial_components / 2
    
    # Find where we hit half the initial number of components
    idx_half = np.where(train_components <= half_components)[0]
    eps_half = train_thresholds[idx_half[0]] if len(idx_half) > 0 else None
    
    # Find where we hit a single component
    idx_single = np.where(train_components <= 1)[0]
    eps_single = train_thresholds[idx_single[0]] if len(idx_single) > 0 else None
    
    # Find key threshold around epsilon=10-15 (critical for G-I tradeoff)
    idx_gi = np.where((train_thresholds >= 10) & (train_thresholds <= 15))[0]
    if len(idx_gi) > 0:
        eps_gi = train_thresholds[idx_gi[0]]
        components_gi = train_components[idx_gi[0]]
    else:
        eps_gi = None
        components_gi = None
    
    # Print analysis
    print("\nG-I Threshold Analysis:")
    print(f"Initial components: {n_initial_components}")
    print(f"50% components lost at epsilon: {eps_half:.2f}" if eps_half else "50% components not reached")
    print(f"Fully connected at epsilon: {eps_single:.2f}" if eps_single else "Not fully connected")
    print(f"At epsilon 10-15 range: {eps_gi:.2f} -> {components_gi} components" if eps_gi else "No data in 10-15 range")
    
    # Plot with both train and OOD data if available
    if ood_thresholds is not None and ood_components is not None:
        plt.figure(figsize=(10, 6))
        
        # Plot train data
        plt.plot(train_thresholds, train_components, 'b-', linewidth=2.5, label='Training Species')
        
        # Plot OOD data
        plt.plot(ood_thresholds, ood_components, 'r--', linewidth=2.5, label='OOD Species')
        
        # Highlight key thresholds
        if eps_half:
            plt.axvline(x=eps_half, color='green', linestyle='--', alpha=0.5)
            plt.text(eps_half, n_initial_components*0.9, f"Half components\nε={eps_half:.2f}", 
                     fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="green", alpha=0.7))
        
        if eps_gi:
            plt.axvline(x=eps_gi, color='purple', linestyle='--', alpha=0.5)
            plt.text(eps_gi, n_initial_components*0.7, f"G-I critical\nε={eps_gi:.2f}", 
                     fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="purple", alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epsilon (ε)', fontsize=14)
        plt.ylabel('Number of Connected Components', fontsize=14)
        plt.title('Connected Components: Training vs. OOD Species', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'train_vs_ood_components_modelsubset.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved train vs. OOD plot to {save_path}")
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze test subset of birds')
    parser.add_argument('--max-dim', type=int, default=1, 
                        help='Maximum homology dimension to compute')
    parser.add_argument('--output-dir', type=str, default='../results/test_subset_analysis',
                        help='Directory to save results')
    parser.add_argument('--highlight-eps', type=float, nargs='+',
                        help='Epsilon values to highlight (e.g., 10 15)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    data = load_bird_data()
    
    # Extract the subsets
    all_species = data['available_species']
    all_distances = data['evo_distances']
    train_species = data['train_species']
    train_distances = data['train_evo_distances']
    ood_species = data['ood_species']
    ood_distances = data['ood_evo_distances']
    
    print(f"\nAnalyzing {len(all_species)} total species")
    print(f"Training subset: {len(train_species)} species")
    print(f"OOD subset: {len(ood_species)} species")
    
    # Run persistent homology on all species
    if USE_RIPSER is not None:
        all_results = run_persistent_homology(all_distances, max_dim=args.max_dim)
        
        # Plot connected components for all species
        all_thresholds, all_components = plot_connected_components(
            all_results['dgms'], 
            all_distances,
            all_species,
            highlight_epsilon=args.highlight_eps,
            output_dir=args.output_dir,
            suffix='_all'
        )
        
        # Plot threshold distribution
        plot_threshold_distribution(
            all_results['dgms'],
            all_distances,
            output_dir=args.output_dir,
            suffix='_all'
        )
        
        # Run persistent homology on training species
        train_results = run_persistent_homology(train_distances, max_dim=args.max_dim)
        
        # Plot connected components for training species
        train_thresholds, train_components = plot_connected_components(
            train_results['dgms'], 
            train_distances,
            train_species,
            highlight_epsilon=args.highlight_eps,
            output_dir=args.output_dir,
            suffix='_train'
        )
        
        # If we have enough OOD species, analyze them too
        if len(ood_species) > 1:
            ood_results = run_persistent_homology(ood_distances, max_dim=args.max_dim)
            
            # Plot connected components for OOD species
            ood_thresholds, ood_components = plot_connected_components(
                ood_results['dgms'], 
                ood_distances,
                ood_species,
                highlight_epsilon=args.highlight_eps,
                output_dir=args.output_dir,
                suffix='_ood'
            )
            
            # Compare train vs OOD
            analyze_gi_thresholds(
                train_thresholds, train_components,
                ood_thresholds, ood_components,
                output_dir=args.output_dir
            )
        else:
            # Just analyze training set if not enough OOD species
            analyze_gi_thresholds(
                train_thresholds, train_components,
                output_dir=args.output_dir
            )
        
        # Plot distance matrices for visualization
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(all_distances, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.title(f'All Species ({len(all_species)})')
        
        plt.subplot(1, 3, 2)
        plt.imshow(train_distances, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.title(f'Training Species ({len(train_species)})')
        
        if len(ood_species) > 1:
            plt.subplot(1, 3, 3)
            plt.imshow(ood_distances, cmap='viridis')
            plt.colorbar(label='Distance')
            plt.title(f'OOD Species ({len(ood_species)})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'distance_matrices_modelsubset.png'), dpi=300)
        plt.close()
    else:
        print("Skipping persistent homology analysis (no library available)")
        
        # Just create simple histograms of distance values
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 3, 1)
        plt.hist(all_distances.flatten(), bins=30, alpha=0.7)
        plt.title(f'All Species ({len(all_species)})')
        plt.xlabel('Evolutionary Distance')
        plt.axvline(x=10, color='red', linestyle='--')
        plt.axvline(x=15, color='green', linestyle='--')
        
        plt.subplot(1, 3, 2)
        plt.hist(train_distances.flatten(), bins=30, alpha=0.7)
        plt.title(f'Training Species ({len(train_species)})')
        plt.xlabel('Evolutionary Distance')
        plt.axvline(x=10, color='red', linestyle='--')
        plt.axvline(x=15, color='green', linestyle='--')
        
        if len(ood_species) > 1:
            plt.subplot(1, 3, 3)
            plt.hist(ood_distances.flatten(), bins=30, alpha=0.7)
            plt.title(f'OOD Species ({len(ood_species)})')
            plt.xlabel('Evolutionary Distance')
            plt.axvline(x=10, color='red', linestyle='--')
            plt.axvline(x=15, color='green', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'distance_histograms_modelsubset.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    main() 