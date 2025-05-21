#!/bin/bash

# Create directory for results
mkdir -p ../data/new_datasplit

# Run the new script with diverse data splits
python3 new_datasplit_birds.py \
  --num-species 15 \
  --batch-size 32 \
  --grad-accum 4 \
  --epochs 20 \
  --alpha 0.5

echo "Experiment complete!" 