# Evil Twins: Prompt Optimization Framework

A modular framework for optimizing prompts using various attack methods including GCG (Gradient-based optimization) and soft prompts.

## Overview

This framework has been refactored from a single large script into a modular, maintainable structure that separates concerns and makes it easy to add new attack methods.

## Project Structure

```
evil_twins/
├── __init__.py              # Main package exports
├── config.py                # Configuration constants and dataclasses
├── utils.py                 # Utility functions (model loading, prompt building)
├── data.py                  # Dataset and data loading functionality
├── model_wrappers.py        # Model wrapper classes for embeddings
├── attacks/                 # Attack algorithms
│   ├── __init__.py
│   ├── gcg.py              # GCG (Gradient-based) optimization
│   └── soft_prompt.py      # Soft prompt optimization
├── evaluation.py            # Evaluation metrics (KL divergence, perplexity)
├── viz.py                  # Visualization functions
├── main.py                 # Command-line interface
├── example.py              # Usage examples
└── README.md               # This file
```

## Key Features

- **Modular Design**: Each component has a single responsibility
- **Configuration Management**: Dataclasses for clean parameter management
- **Multiple Attack Methods**: GCG and soft prompt optimization
- **Evaluation Metrics**: KL divergence, perplexity, and more
- **Visualization**: Training curves and comparison plots
- **CLI Interface**: Easy command-line usage
- **Extensible**: Easy to add new attack methods

## Installation

```bash
# Install dependencies
pip install torch transformers einops tqdm matplotlib numpy
```

## Quick Start

### Command Line Usage

```bash
# Run GCG optimization
python -m evil_twins.main \
    --model_name "EleutherAI/pythia-14m" \
    --attack_type gcg \
    --orig_prompt "Tell me a recipe." \
    --optim_prompt "!!!!!" \
    --n_epochs 10 \
    --output_dir ./results

# Run soft prompt optimization
python -m evil_twins.main \
    --model_name "EleutherAI/pythia-14m" \
    --attack_type soft \
    --orig_prompt "Write a story about a cat." \
    --optim_prompt "!!!" \
    --n_epochs 10 \
    --learning_rate 1e-3 \
    --output_dir ./results
```

### Programmatic Usage

```python
import torch
from evil_twins import (
    load_model_tokenizer, DocDataset, 
    GCGConfig, optim_gcg, plot_training_curves
)

# Load model
model, tokenizer = load_model_tokenizer("EleutherAI/pythia-14m")

# Create dataset
dataset = DocDataset(
    model=model,
    tokenizer=tokenizer,
    orig_prompt="Tell me a recipe.",
    optim_prompt="!" * 5,
    n_docs=10,
    doc_len=20
)

# Configure and run GCG
config = GCGConfig(n_epochs=10, batch_size=5)
results, best_prompt = optim_gcg(
    model, tokenizer, dataset, config, "results.json"
)

# Plot results
plot_training_curves("results.json", "curves.png")
```

## Configuration

The framework uses dataclasses for clean configuration management:

### GCGConfig
```python
@dataclass
class GCGConfig:
    n_epochs: int = 100
    kl_every: int = 10
    batch_size: int = 10
    top_k: int = 256
    gamma: float = 0.0
    early_stop_kl: float = 0.0
    suffix_mode: bool = False
```

### SoftPromptConfig
```python
@dataclass
class SoftPromptConfig:
    n_epochs: int = 100
    kl_every: int = 10
    batch_size: int = 10
    learning_rate: float = 1e-3
```

## Attack Methods

### GCG (Gradient-based Optimization)
- Optimizes hard prompts by replacing tokens based on gradients
- Uses top-k token selection for efficient optimization
- Supports fluency penalty with gamma parameter
- Can run in suffix mode for single-document optimization

### Soft Prompt Optimization
- Optimizes continuous embeddings instead of discrete tokens
- Uses gradient descent with Adam optimizer
- Supports custom learning rates and batch sizes
- Saves best embeddings to disk

## Evaluation Metrics

- **KL Divergence**: Measures difference between original and optimized prompt distributions
- **Perplexity**: Measures model performance on generated documents
- **Loss Tracking**: Training and validation loss curves

## Visualization

The framework provides several plotting functions:

- `plot_training_curves()`: Loss, KL divergence, and NLL curves
- `plot_prompt_comparison()`: Original vs optimized prompt evolution
- `plot_kl_convergence()`: Compare multiple experiments

## Adding New Attack Methods

To add a new attack method:

1. Create a new file in `attacks/` (e.g., `attacks/zeroth_order.py`)
2. Define a configuration dataclass in `config.py`
3. Implement the optimization function with clear interface
4. Add to `attacks/__init__.py` exports
5. Update `main.py` CLI to support the new method

Example:
```python
# attacks/zeroth_order.py
def optim_nes(model, dataset, config, log_fpath):
    """NES (Natural Evolution Strategy) optimization."""
    # Implementation here
    pass

# config.py
@dataclass
class NESConfig:
    n_epochs: int = 100
    population_size: int = 50
    sigma: float = 0.1
```

## Testing

The modular structure makes it easy to test individual components:

```python
# Test dataset generation
from evil_twins.data import DocDataset
dataset = DocDataset(...)
assert len(dataset) > 0

# Test evaluation metrics
from evil_twins.evaluation import compute_dataset_kl
kl, std = compute_dataset_kl(model, dataset, batch_size=5)
assert kl >= 0
```

## Contributing

1. Follow the modular structure
2. Use dataclasses for configuration
3. Add type hints to all functions
4. Include docstrings for all public APIs
5. Add tests for new functionality

## License

This project is licensed under the MIT License. 