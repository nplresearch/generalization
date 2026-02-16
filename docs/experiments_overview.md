# Experiments Overview

This document summarizes all four experimental paradigms and how they relate to the theoretical predictions.

## Theoretical Framework

The G-I tradeoff is parameterized by **semantic resolution** $\varepsilon$. A similarity function with resolution $\varepsilon$ can only distinguish stimuli within distance $\varepsilon$ of each other. Letting $b(\varepsilon)$ denote the fraction of stimulus space covered by a ball of radius $\varepsilon$:

**Constant similarity model (Theorem 1):**
$$p_S = \tfrac{1}{2} + b - b^2, \qquad p_I = 1 - \tfrac{b}{2}$$

**Linear decay model (Proposition 1):**
$$p_S = \tfrac{1}{2} + b - (1.5 - \ln 2)\,b^2, \qquad p_I = 1 - (1 - \ln 2)\,b$$

These define Pareto fronts in $(p_I, p_S)$ space. No system — regardless of architecture — can exceed these bounds.

**Multi-item scaling:**
$$p_I^n \approx \frac{1}{b \cdot n}$$

Identification accuracy collapses as $1/n$ with the number of items, explaining Miller's "7 plus or minus 2" limit.

---

## Experiment 1: Toy Model (Synthetic)

**Goal:** Verify theoretical predictions in a controlled setting with known metric structure.

**Setup:**
- Stimuli are categorical items on a flat circle or multi-slot composites
- Configurable distance matrices control the ground-truth similarity structure
- Models: single linear projection (TransformerModel) vs. per-slot projections (CompositionalModel)
- Training objective: similarity loss (NLL) + reconstruction loss (MSE)

**Key findings:**
- ReLU networks spontaneously develop resolution boundaries during training
- Training trajectories in $(p_S, p_I)$ space track the theoretical Pareto front
- Compositional (slot-based) models can partially escape the tradeoff

**Notebooks:**
- `Toy_model/notebooks/clean_experiment.ipynb` — Main experimental setup
- `Toy_model/notebooks/slot_model_clean.ipynb` — Slot vs. non-slot comparison

---

## Experiment 2: CNN Birds (CUB-200)

**Goal:** Test whether the tradeoff holds in a realistic vision task with natural similarity structure.

**Setup:**
- 200 bird species from CUB-200-2011
- Phylogenetic (evolutionary) distance as ground-truth similarity metric
- ResNet-50 fine-tuned with dual loss: $\mathcal{L} = (1-\alpha)\mathcal{L}_\text{ID} + \alpha\mathcal{L}_\text{Gen}$
- Threshold sweep over $\varepsilon$ traces empirical G-I curves per alpha value

**Key findings:**
- Empirical G-I curves match theoretical Pareto fronts
- $\alpha$ controls where models land on the front: low $\alpha$ favors identification, high $\alpha$ favors generalization
- The tradeoff is robust across random seeds

**Guide:** See [cnn_birds_guide.md](cnn_birds_guide.md) for detailed instructions.

---

## Experiment 3: VLM Shapes

**Goal:** Probe whether pre-trained Vision-Language Models exhibit the predicted resolution limits.

**Setup:**
- Models: Gemma-2-2b-it, Qwen2.5-7B-Instruct
- Tasks: color similarity/identification, number discrimination, spatial position resolution
- Stimuli are programmatically generated visual scenes presented as images

**Key findings:**
- VLMs show clear resolution boundaries — accuracy is high within $\varepsilon$, then drops to chance
- Binding errors (confusing which color goes with which shape) are consistent with distributed representations
- Different tasks (color, position, number) reveal different resolution scales

**Notebooks:**
- `VLM_shapes/Color_similarity_identification.ipynb`
- `VLM_shapes/Number_similarity_identification.ipynb`
- `VLM_shapes/Stencil_position.ipynb`
- `VLM_shapes/VLM_visualizations.ipynb`

---

## Experiment 4: LLM Dates

**Goal:** Test whether LLMs exhibit temporal resolution limits on date-based reasoning.

**Setup:**
- Models: Gemma-2-2b-it, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct
- Tasks: year-based similarity judgments and identification
- Resolution probed by varying the temporal gap between comparison stimuli

**Key findings:**
- LLMs exhibit ~70–80 year temporal resolution
- Similarity accuracy degrades smoothly beyond the resolution boundary
- The pattern is universal across model families

**Notebook:** `LLM_dates/LLM_analysis.ipynb`

---

## Summary Table

| Experiment | Domain | Models | Similarity Metric | Key Parameter |
|------------|--------|--------|-------------------|---------------|
| Toy Model | Synthetic | Linear / Compositional | Configurable matrix | `lambda_sim`, `lambda_rec` |
| CNN Birds | Vision (200 species) | ResNet-50 | Phylogenetic distance | $\alpha \in [0, 1]$ |
| VLM Shapes | Vision-Language | Gemma, Qwen | Color/position/number distance | (probing, no training) |
| LLM Dates | Language | Gemma, Llama, Qwen | Temporal distance (years) | (probing, no training) |
