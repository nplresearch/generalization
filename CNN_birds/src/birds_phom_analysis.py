#!/usr/bin/env python3
"""
Persistent Homology Analysis for Bird Species Metric Space
"""
import dendropy
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
import pickle

# Import persistent homology libraries (choose one)
try:
    import ripser  # Ripser.py is a lightweight option
    USE_RIPSER = True
except ImportError:
    try:
        import gudhi  # GUDHI is more comprehensive
        USE_RIPSER = False
    except ImportError:
        raise ImportError("Please install either ripser or gudhi for persistent homology analysis")

def log_step(message):
    """Print timestamped log message"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_data():
    """Load the necessary data for the analysis"""
    log_step("Loading data...")
    
    # Check if we can load cached data
    cache_file = "../data/phom_data_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            log_step("Loaded data from cache")
            return data
    
    # Load and parse the Newick tree file
    log_step("Loading Newick tree file...")
    tree = dendropy.Tree.get(path="../data/birds_species.nwk", schema="newick")
    log_step("Tree file loaded successfully")
    
    # Load CUB dataset bird names
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
    
    # Load mapping between common and scientific names
    # This is a simplified version of the code in the main script
    common_to_scientific = {}
    
    # Add some key mappings
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
    
    # Find additional mappings
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
    
    # Try to load feature-based distances from a saved file
    if os.path.exists("../data/feature_distances.npy"):
        log_step("Loading pre-computed feature distances...")
        feature_distances = np.load("../data/feature_distances.npy")
    else:
        log_step("No pre-computed feature distances found.")
        feature_distances = None
    
    # Create a dictionary with all the data we need
    data = {
        'tree': tree,
        'class_names': class_names,
        'common_to_scientific': common_to_scientific,
        'available_species': available_species,
        'evo_distances': evo_distances,
        'feature_distances': feature_distances
    }
    
    # Cache the data for future use
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    log_step("Data cached for future use")
    
    return data

def run_persistent_homology(distance_matrix, max_dim=1):
    """
    Compute persistent homology on the distance matrix
    
    Args:
        distance_matrix: Square distance matrix
        max_dim: Maximum homology dimension to compute
    
    Returns:
        Dictionary with persistent homology results
    """
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
            'cocycles': result.get('cocycles', None),  # Representative cocycles
            'num_edges': result.get('num_edges', None)  # Number of edges in the Vietoris-Rips complex
        }
    else:
        # Using GUDHI
        # Convert distance matrix to lower triangular form
        dist_vector = squareform(distance_matrix)
        
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
            'dgms': [np.array(dgm) for dgm in dgms],
            'num_simplices': simplex_tree.num_simplices()
        }

def plot_persistence_diagram(diagrams, title, save_path=None):
    """
    Plot persistence diagrams
    
    Args:
        diagrams: List of persistence diagrams, one for each dimension
        title: Title for the plot
        save_path: If provided, save plot to this path
    """
    plt.figure(figsize=(10, 8))
    
    # Get maximum death time for plot limits
    max_death = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            max_death = max(max_death, np.max(dgm[:, 1]))
    
    # Plot diagonal line
    plt.plot([0, max_death*1.1], [0, max_death*1.1], 'k--', alpha=0.5)
    
    # Plot points for each dimension
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        
        # Get finite points (remove inf)
        finite = ~np.isinf(dgm[:, 1])
        inf_points = ~finite
        
        # Plot finite points
        if np.any(finite):
            plt.scatter(
                dgm[finite, 0], dgm[finite, 1], 
                s=50, c=colors[dim % len(colors)], marker=markers[dim % len(markers)],
                alpha=0.8, label=f'H{dim} (finite)'
            )
        
        # Plot points with infinite death separately
        if np.any(inf_points):
            # Replace inf with max_death * 1.05 for visualization
            inf_y = max_death * 1.05
            plt.scatter(
                dgm[inf_points, 0], np.ones(np.sum(inf_points)) * inf_y,
                s=50, c=colors[dim % len(colors)], marker=markers[dim % len(markers)],
                alpha=0.5, edgecolor='black', label=f'H{dim} (inf)'
            )
    
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', alpha=0.3)
    plt.axvline(x=0, color='k', alpha=0.3)
    
    plt.xlabel('Birth', fontsize=14)
    plt.ylabel('Death', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper left')
    
    # Add text with persistence information
    total_persistence = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        # Calculate total persistence (sum of death - birth)
        finite = ~np.isinf(dgm[:, 1])
        if np.any(finite):
            persistence = np.sum(dgm[finite, 1] - dgm[finite, 0])
            plt.annotate(f"H{dim} persistence: {persistence:.2f}",
                        xy=(0.02, 0.98 - 0.05*dim),
                        xycoords='axes fraction',
                        fontsize=10)
            total_persistence += persistence
    
    plt.annotate(f"Total persistence: {total_persistence:.2f}",
                xy=(0.02, 0.98 - 0.05*(len(diagrams)+1)),
                xycoords='axes fraction',
                fontsize=10,
                fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved persistence diagram to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_barcode(diagrams, title, save_path=None, max_points=50):
    """
    Plot persistence barcodes
    
    Args:
        diagrams: List of persistence diagrams, one for each dimension
        title: Title for the plot
        save_path: If provided, save plot to this path
        max_points: Maximum number of points to show per dimension
    """
    plt.figure(figsize=(12, 8))
    
    # Get maximum death time for plot limits
    max_death = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            # Filter out infinite values
            finite_death = dgm[~np.isinf(dgm[:, 1]), 1]
            if len(finite_death) > 0:
                max_death = max(max_death, np.max(finite_death))
    
    # Create list of all finite death values for percentile calculation
    all_finite_deaths = []
    for dgm in diagrams:
        if len(dgm) > 0:
            # Filter out infinite values
            finite_death = dgm[~np.isinf(dgm[:, 1]), 1]
            if len(finite_death) > 0:
                all_finite_deaths.append(finite_death)
    
    # Limit upper bound for plot to exclude extreme outliers
    if all_finite_deaths:
        max_plot = np.percentile(np.concatenate(all_finite_deaths), 98)
    else:
        # Fallback if no finite death values
        max_plot = max_death if max_death > 0 else 10.0
    
    # Number of bars
    total_bars = sum(min(len(dgm), max_points) for dgm in diagrams)
    if total_bars == 0:
        print("No persistence pairs to plot")
        return
    
    # Y-values for bars
    bar_height = 0.8
    y_values = []
    bar_labels = []
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    current_y = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        
        # Sort by persistence (death - birth)
        persistence = dgm[:, 1] - dgm[:, 0]
        sorted_indices = np.argsort(-persistence)  # Sort in descending order
        
        # Limit number of points to plot
        num_points = min(len(dgm), max_points)
        indices = sorted_indices[:num_points]
        
        # For each point, plot a horizontal bar
        for i, idx in enumerate(indices):
            birth, death = dgm[idx, 0], dgm[idx, 1]
            if np.isinf(death):
                # Use plot limit for infinite bars
                death = max_plot * 1.1
            
            # Plot the bar
            plt.plot([birth, death], [current_y, current_y], '-', 
                     linewidth=bar_height, color=colors[dim % len(colors)], alpha=0.7)
            
            # Add point at birth
            plt.plot(birth, current_y, 'o', color=colors[dim % len(colors)], 
                     markersize=5, alpha=0.8)
            
            # Add point at death (only for finite death)
            if not np.isinf(dgm[idx, 1]):
                plt.plot(death, current_y, 'x', color=colors[dim % len(colors)], 
                         markersize=5, alpha=0.8)
            
            y_values.append(current_y)
            bar_labels.append(f"H{dim}_{i}")
            current_y += 1
    
    # Set y-axis limits and ticks
    plt.ylim(-1, current_y)
    plt.yticks([])  # Hide y ticks
    
    # Add dimension labels
    dim_labels = []
    dim_positions = []
    current_y = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        
        num_points = min(len(dgm), max_points)
        dim_positions.append(current_y + num_points/2)
        dim_labels.append(f"H{dim}")
        current_y += num_points
    
    plt.yticks(dim_positions, dim_labels, fontsize=12)
    
    # Set x-axis limits - ensure we don't have inf or nan values
    plt.xlim(0, max_plot*1.1)
    
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlabel('Epsilon (ε)', fontsize=14)
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved barcode to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_connected_components(diagrams, distance_matrix, species_names, output_dir=None):
    """
    Analyze the formation of connected components
    
    Args:
        diagrams: List of persistence diagrams, one for each dimension
        distance_matrix: Distance matrix used for PHom
        species_names: List of species names corresponding to distance_matrix
        output_dir: Directory to save output plots
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
        
        plt.vlines(x=21.0, ymin=0, ymax=max(components), colors='red', linestyles='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'connected_components_allbirds.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved connected components plot to {save_path}")
        plt.close()
        
        # Analyze the most important mergers (largest persistence H0 features)
        print("\nMost significant connected component merges:")
        
        # Sort H0 features by persistence
        h0_persistence = h0_diagram[finite, 1] - h0_diagram[finite, 0]
        sorted_indices = np.argsort(-h0_persistence)  # Sort in descending order
        
        top_n = min(5, len(sorted_indices))
        for i in range(top_n):
            idx = sorted_indices[i]
            birth, death = h0_diagram[idx, 0], h0_diagram[idx, 1]
            print(f"{i+1}. Epsilon = {death:.4f}, Persistence = {death-birth:.4f}")
            
            # Find which components merged at this epsilon
            # This is a simplification - for precise analysis we'd need to track
            # the full persistent homology computation
            adj_matrix_before = distance_matrix <= (death - 0.001)
            adj_matrix_after = distance_matrix <= (death + 0.001)
            np.fill_diagonal(adj_matrix_before, 0)
            np.fill_diagonal(adj_matrix_after, 0)
            
            # TODO: More advanced analysis of which components merged could be added here

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Persistent Homology Analysis for Bird Species')
    parser.add_argument('--max-dim', type=int, default=1, 
                        help='Maximum homology dimension to compute')
    parser.add_argument('--output-dir', type=str, default='../results/phom_analysis',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    data = load_data()
    
    # Run persistent homology on evolutionary distances
    evo_results = run_persistent_homology(data['evo_distances'], max_dim=args.max_dim)
    
    # Plot persistence diagram for evolutionary distances
    plot_persistence_diagram(
        evo_results['dgms'], 
        'Persistence Diagram - Evolutionary Distances',
        save_path=os.path.join(args.output_dir, 'evo_persistence_diagram_allbirds.png')
    )
    
    # Plot barcode for evolutionary distances
    plot_barcode(
        evo_results['dgms'], 
        'Persistence Barcode - Evolutionary Distances',
        save_path=os.path.join(args.output_dir, 'evo_barcode_allbirds.png')
    )
    
    # Analyze connected components
    analyze_connected_components(
        evo_results['dgms'], 
        data['evo_distances'],
        data['available_species'],
        output_dir=args.output_dir
    )
    
    # If we have feature distances, analyze those too
    if data['feature_distances'] is not None:
        feature_results = run_persistent_homology(data['feature_distances'], max_dim=args.max_dim)
        
        # Plot persistence diagram for feature distances
        plot_persistence_diagram(
            feature_results['dgms'], 
            'Persistence Diagram - Feature Distances',
            save_path=os.path.join(args.output_dir, 'feature_persistence_diagram_allbirds.png')
        )
        
        # Plot barcode for feature distances
        plot_barcode(
            feature_results['dgms'], 
            'Persistence Barcode - Feature Distances',
            save_path=os.path.join(args.output_dir, 'feature_barcode_allbirds.png')
        )
        
        # Compare the two distance matrices
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(data['evo_distances'], cmap='viridis')
        plt.colorbar(label='Distance')
        plt.title('Evolutionary Distances')
        
        plt.subplot(1, 2, 2)
        plt.imshow(data['feature_distances'], cmap='viridis')
        plt.colorbar(label='Distance')
        plt.title('Feature Distances')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'distance_matrices_comparison_allbirds.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    main() 