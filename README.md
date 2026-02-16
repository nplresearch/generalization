# Bound by Semanticity: The Generalization-Identification Tradeoff

Code and experiments for [**"Bound by Semanticity: Universal Laws Governing the Generalization-Identification Tradeoff"**](https://arxiv.org/abs/2506.14797) (Nurisso, Fernando, Deshpande, Perotti, Marjieh, Frankland, Lewis, Webb, Campbell, Vaccarino, Cohen & Petri, 2025), which derives exact Pareto frontiers for the tradeoff between generalization and identification in neural representations.

> **The same property that makes neural networks powerful — shared, distributed representations — imposes hard limits on what they can do.**

## Key Result

Any system using distributed representations faces a fundamental tradeoff: representations that support **generalization** (recognizing similarity structure) necessarily degrade **identification** (distinguishing individual items). For a similarity function with resolution $\varepsilon$, the achievable performance is bounded by:

$$p_S = \tfrac{1}{2} + b(\varepsilon) - b(\varepsilon)^2, \qquad p_I = 1 - \tfrac{b(\varepsilon)}{2}$$

where $b(\varepsilon)$ is the fraction of stimulus space within the resolution ball. These equations define a **Pareto front** that no system — regardless of architecture or scale — can exceed.

## Repository Structure

```
├── CNN_birds/          # Bird species experiments (ResNet-50 + CUB-200)
├── Toy_model/          # Controlled synthetic experiments
├── VLM_shapes/         # Vision-Language Model experiments
├── LLM_dates/          # Large Language Model temporal reasoning
├── website/            # Interactive web explainers
└── docs/               # API and experiment documentation
```

### CNN_birds — Real-World Validation with Bird Species

Trains ResNet-50 on CUB-200-2011 bird species with a dual loss controlled by parameter $\alpha \in [0,1]$:

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_\text{ID} + \alpha\,\mathcal{L}_\text{Gen}$$

where $\mathcal{L}_\text{ID}$ is cross-entropy classification loss and $\mathcal{L}_\text{Gen}$ aligns embedding distances with phylogenetic (evolutionary) distances between species.

**Key files:**
- `src/birdsbirdsbirds.py` — Training and evaluation pipeline (~1800 lines)
- `src/create_pickle_files.py` — Aggregates results across runs
- `notebooks/figure.ipynb` — Main paper figures
- `notebooks/supp.ipynb` — Supplementary analyses

### Toy_model — Controlled Synthetic Experiments

Validates theoretical predictions using minimal models on synthetic stimuli with known metric structure. Compares **non-compositional** (single linear projection) and **compositional** (per-slot embeddings with max-over-slots similarity) architectures.

**Key files:**
- `src/MillerDataGen.py` — Data generators for flat and multi-slot stimuli
- `src/MillerTest.py` — `TransformerModel`, `CompositionalModel`, and `MillerTestEvaluator`
- `src/MillerTrainer.py` — Training loop with similarity + reconstruction loss
- `notebooks/` — Slot vs. non-slot comparison experiments

### VLM_shapes — Vision-Language Model Probing

Tests pre-trained VLMs (Gemma-2-2b-it, Qwen2.5-7B-Instruct) on color similarity, number discrimination, and spatial resolution tasks. Pre-computed results in `data/`.

### LLM_dates — LLM Temporal Reasoning

Probes LLMs (Gemma, Llama-3.2, Qwen2.5) on year-of-birth similarity and identification. Models exhibit ~70–80 year temporal resolution. Pre-computed results in `saved_results/`.

### website — Interactive Explainers

D3.js + KaTeX web pages with interactive Pareto front visualizations and training trajectory animations. Three variants: `explainer-npl.html` (NPL-branded), `explainer_1.html` / `explainer_2.html` (standalone).

## Quick Start

### Environment Setup

```bash
conda env create -f CNN_birds/env.yml
conda activate miller_law
```

Dependencies: Python 3.9+, PyTorch 2.1+, torchvision, dendropy, scikit-learn, matplotlib, seaborn, pandas, scipy.

### Running CNN Experiments

```bash
cd CNN_birds

# Train across alpha values with multiple seeds
python src/birdsbirdsbirds.py --alpha 0.0 0.25 0.5 0.75 1.0 --epochs 15 --num-seeds 5

# Aggregate results (edit DATE_PATTERNS first)
python src/create_pickle_files.py

# Generate figures
jupyter notebook notebooks/figure.ipynb
```

**CLI arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--alpha` | G-I tradeoff weight(s), space-separated | `0.5` |
| `--batch-size` | Samples per batch | `8` |
| `--grad-accum` | Gradient accumulation steps | `4` |
| `--epochs` | Training epochs | `15` |
| `--num-seeds` | Number of random seeds | `5` |
| `--seeds` | Explicit seed list (overrides `--num-seeds`) | — |

### Running Toy Model Experiments

Open and run the notebooks in `Toy_model/notebooks/`. The Python modules in `src/` are imported directly by the notebooks:

```python
from MillerDataGen import MillerDataGenerator, DataGeneratorConfig
from MillerTest import TransformerModel, MillerTestConfig, MillerTestEvaluator
from MillerTrainer import MillerTrainer, TrainerConfig

config = DataGeneratorConfig(num_inputs=32, sequence_len=3)
data_gen = MillerDataGenerator(config)
model = TransformerModel(MillerTestConfig(input_dim=32, feature_dim=32))
trainer = MillerTrainer(model, data_gen, TrainerConfig(num_epochs=100))
history = trainer.train()
```

### Viewing VLM / LLM Results

Results are pre-computed. Open the Jupyter notebooks directly:

```bash
jupyter notebook VLM_shapes/VLM_visualizations.ipynb
jupyter notebook LLM_dates/LLM_analysis.ipynb
```

## Data Requirements

- **CUB-200-2011**: Download from [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) and place in `CNN_birds/CUB_200_2011/`
- **Phylogenetic tree**: Included at `CNN_birds/data/birds_species.nwk` (Newick format)
- **VLM/LLM results**: Pre-computed and included in-repo

## Citation

```bibtex
@article{nurisso2025bound,
  title={Bound by Semanticity: Universal Laws Governing the Generalization-Identification Tradeoff},
  author={Nurisso, Marco and Fernando, Jesseba and Deshpande, Raj and Perotti, Alan and Marjieh, Raja and Frankland, Steven M. and Lewis, Richard L. and Webb, Taylor W. and Campbell, Declan and Vaccarino, Francesco and Cohen, Jonathan D. and Petri, Giovanni},
  journal={arXiv preprint arXiv:2506.14797},
  year={2025}
}
```

## Related Work

- Frankland, Webb, Lewis & Cohen (2025). [No Coincidence, George: Processing Limits as the Curse of Generalization](https://osf.io/preprints/psyarxiv/cjuxb_v2). *PsyArXiv*.
- Campbell et al. (2024). [Understanding the Limits of VLMs Through the Lens of the Binding Problem](https://arxiv.org/abs/2411.00238). *NeurIPS*.
- Musslick & Cohen (2021). [Rationalizing constraints on the capacity for cognitive control](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(21)00148-0). *TiCS*.
- Webb et al. (2020). [Learning Representations that Support Extrapolation](https://proceedings.mlr.press/v119/webb20a.html). *ICML*.

## License

Please see the individual dataset licenses (CUB-200-2011) for data usage terms.
