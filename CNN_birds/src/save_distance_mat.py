import dendropy
import datetime
import os
import numpy as np

# Add diagnostic function to track progress 
def log_step(message):
    """Print timestamped log message"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

log_step("Starting distance matrix calculation")

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

# Load CUB dataset and build mapping
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

# Code to map common names to scientific names
common_to_scientific = {}

# Your mapping code here - include the code from the script for automatic mapping
# ...

# Manual mappings - use the same mappings as in your script
manual_mappings = {
    'Black_footed_Albatross': 'Phoebastria nigripes',
    'Laysan_Albatross': 'Phoebastria immutabilis',
    'Sooty_Albatross': 'Phoebetria fusca',
    # Add all your manual mappings here
}

# Apply manual mappings and validate as in your script
# ...

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
distance_matrix = np.zeros((species_count, species_count))

for i, sp1 in enumerate(available_species):
    for j, sp2 in enumerate(available_species):
        if i == j:
            continue
        sci1 = common_to_scientific[sp1]
        sci2 = common_to_scientific[sp2]
        distance = get_evolutionary_distance(tree, sci1, sci2)
        if distance is not None:
            distance_matrix[i, j] = distance
log_step("Evolutionary distance matrix built")

# Save the matrix for future use
np.save("/home/jefe/repos/generalization_transformer/data/evolutionary_distances.npy", distance_matrix)
log_step("Saved distance matrix to file")