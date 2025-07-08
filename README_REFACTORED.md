# Evil Twins: Refactored Prompt Optimization Package

This is a production-ready, object-oriented refactoring of the original `evil_twins` package. The code has been restructured to follow software engineering best practices while maintaining full backward compatibility.

## 🚀 Key Improvements

### **Object-Oriented Design**
- **Modular Architecture**: Separated concerns into distinct modules
- **Class-Based APIs**: Intuitive object-oriented interfaces
- **Inheritance Hierarchy**: Clean abstraction layers for different optimization strategies

### **Production-Ready Features**
- **Comprehensive Logging**: Structured logging throughout the codebase
- **Error Handling**: Robust exception handling and validation
- **Configuration Management**: Centralized configuration with validation
- **Type Hints**: Complete type annotations for better IDE support
- **Documentation**: Extensive docstrings and examples

### **Enhanced Functionality**
- **Checkpointing**: Save and restore optimization state
- **Visualization**: Advanced plotting and analysis tools
- **Extensibility**: Easy to add new optimization strategies
- **Testing**: Comprehensive test suite

## 📁 Project Structure

```
evil_twins/
├── __init__.py              # Main package interface
├── config.py               # Configuration management
├── models.py               # Model and tokenizer management
├── prompts.py              # Prompt template management
├── datasets.py             # Dataset classes
├── utils.py                # Utility functions
├── visualization.py        # Plotting and visualization
└── optimizers/            # Optimization strategies
    ├── __init__.py
    ├── base.py            # Base optimizer interface
    ├── gcg.py             # GCG hard prompt optimizer
    └── soft.py            # Soft prompt optimizer
```

## 🛠️ Installation

```bash
# Install from source
pip install -e .[all]

# Or install specific components
pip install -e .[dev]      # Development dependencies
pip install -e .[transfer] # Transfer learning dependencies
```

## 📖 Usage Examples

### **Modern Object-Oriented API**

#### **GCG Hard Prompt Optimization**

```python
from evil_twins import ModelManager, DocDataset, GCGOptimizer, OptimizationConfig

# Load model and tokenizer
model_manager = ModelManager("gpt2", use_flash_attn_2=False)
model, tokenizer = model_manager.load_model_and_tokenizer()

# Create dataset
dataset = DocDataset(
    model=model,
    tokenizer=tokenizer,
    orig_prompt="Tell me a recipe for chocolate cake.",
    optim_prompt="! " * 10,  # Initial optimization prompt
    n_docs=50,               # Number of documents to generate
    doc_len=32,              # Length of each document
    gen_batch_size=10,       # Batch size for generation
)

# Configure optimization
config = OptimizationConfig(
    batch_size=8,
    top_k=256,
    gamma=0.0,               # Fluency penalty
    early_stop_kl=0.1,       # Early stopping threshold
)

# Create and run optimizer
optimizer = GCGOptimizer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=config,
    log_fpath="gcg_optimization.json",
)

# Run optimization
results, best_prompt = optimizer.optimize(n_epochs=100, kl_every=5)

# Get results
print(f"Optimized prompt: {optimizer.get_optimized_prompt()}")
print(f"Best KL divergence: {optimizer.best_kl:.4f}")
```

#### **Soft Prompt Optimization**

```python
from evil_twins import SoftPromptOptimizer

# Create soft prompt optimizer
soft_optimizer = SoftPromptOptimizer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=OptimizationConfig(
        batch_size=16,
        learning_rate=1e-3,
    ),
    log_fpath="soft_optimization.json",
    emb_save_fpath="soft_embeddings.pt",
)

# Run optimization
results, best_embeddings = soft_optimizer.optimize(n_epochs=200, kl_every=10)

# Decode embeddings back to text
decoded_prompt = soft_optimizer.decode_embeddings(best_embeddings)
print(f"Decoded prompt: {decoded_prompt}")
```

#### **Visualization and Analysis**

```python
from evil_twins import TrainingVisualizer, create_comprehensive_visualization

# Create visualizer from results
visualizer = TrainingVisualizer(results)

# Plot training curves
visualizer.plot_training_curves("training_curves.png")

# Create comprehensive visualization
create_comprehensive_visualization(
    "optimization_log.json", 
    "visualization_output"
)

# Generate summary report
visualizer.create_summary_report("optimization_summary.txt")
```

### **Legacy API (Backward Compatible)**

The original API is still fully supported:

```python
from evil_twins import load_model_tokenizer, DocDataset, optim_gcg, optim_soft

# Load model and tokenizer
model, tokenizer = load_model_tokenizer("gpt2")

# Create dataset
dataset = DocDataset(
    model=model,
    tokenizer=tokenizer,
    orig_prompt="Tell me a recipe.",
    optim_prompt="! " * 5,
    n_docs=10,
    doc_len=16,
)

# Run GCG optimization
results, ids = optim_gcg(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    n_epochs=50,
    kl_every=5,
    log_fpath="legacy_gcg.json",
    batch_size=8,
    top_k=256,
    gamma=0.0,
)

# Run soft optimization
results, embs = optim_soft(
    model=model,
    dataset=dataset,
    n_epochs=100,
    kl_every=10,
    learning_rate=1e-3,
    log_fpath="legacy_soft.json",
    emb_save_fpath="legacy_embeddings.pt",
    batch_size=16,
)
```

## 🔧 Configuration

### **OptimizationConfig**

Centralized configuration for all optimization parameters:

```python
from evil_twins import OptimizationConfig

config = OptimizationConfig(
    # GCG parameters
    batch_size=16,
    top_k=256,
    gamma=0.0,              # Fluency penalty
    early_stop_kl=0.1,      # Early stopping threshold
    
    # Soft prompt parameters
    learning_rate=1e-3,
    
    # General parameters
    gen_batch_size=10,
    validate_prompt=True,
)

# Convert to/from dictionary
config_dict = config.to_dict()
config_from_dict = OptimizationConfig.from_dict(config_dict)
```

### **ModelConfig**

Manage model-specific prompt templates:

```python
from evil_twins import ModelConfig

# Get template for a model
template = ModelConfig.get_template("gpt2")
print(f"Prefix: {template.prefix}")
print(f"Suffix: {template.suffix}")

# Add custom model mapping
ModelConfig.add_model_mapping("my-model", "vicuna")

# Add custom template
from evil_twins.config import PromptTemplate
custom_template = PromptTemplate(
    prefix="<|system|>",
    suffix="<|user|>",
    description="Custom template"
)
ModelConfig.add_template("custom", custom_template)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_refactored.py

# Run specific tests
python -c "
from test_refactored import test_gcg_optimization, test_soft_optimization
test_gcg_optimization()
test_soft_optimization()
"
```

## 📊 Advanced Features

### **Checkpointing**

Save and restore optimization state:

```python
# Save checkpoint
optimizer.save_checkpoint("checkpoint.json")

# Load checkpoint
optimizer.load_checkpoint("checkpoint.json")
```

### **Custom Optimization Strategies**

Extend the base optimizer for custom strategies:

```python
from evil_twins.optimizers.base import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    def _step(self):
        # Implement custom optimization step
        return {"loss": 0.0, "custom_metric": 1.0}
    
    def optimize(self, n_epochs, kl_every=1):
        # Implement custom optimization loop
        pass
```

### **Model Management**

Advanced model loading and management:

```python
from evil_twins import ModelManager

# Context manager for automatic cleanup
with ModelManager("gpt2") as (model, tokenizer):
    # Use model and tokenizer
    pass

# Manual management
model_manager = ModelManager("gpt2", dtype=torch.float16)
model = model_manager.load_model()
tokenizer = model_manager.load_tokenizer()
```

## 🔍 Monitoring and Debugging

### **Logging**

Structured logging throughout the codebase:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Logs will show:
# - Model loading progress
# - Optimization steps
# - Performance metrics
# - Error messages
```

### **Progress Tracking**

Real-time progress monitoring:

```python
# Progress bars are automatically shown during optimization
# Custom progress tracking
for epoch in range(n_epochs):
    step_info = optimizer._step()
    print(f"Epoch {epoch}: Loss = {step_info['loss']:.4f}")
```

## 🚀 Performance Optimization

### **Memory Management**

- Automatic device placement
- Gradient checkpointing support
- Efficient batch processing

### **Speed Optimizations**

- Flash attention 2 support
- Optimized data loading
- Parallel document generation

## 📈 Results Analysis

### **Training Curves**

Automatically generated plots showing:
- Loss progression
- KL divergence
- Prompt negative log likelihood
- Metric comparisons

### **Summary Reports**

Comprehensive text reports including:
- Final metrics
- Configuration summary
- Prompt evolution
- Performance statistics

## 🤝 Contributing

The refactored codebase is designed for easy extension:

1. **Add new optimizers**: Extend `BaseOptimizer`
2. **Add new models**: Extend `ModelConfig`
3. **Add new visualizations**: Extend `TrainingVisualizer`
4. **Add new datasets**: Extend `DocDataset`

## 📚 API Reference

### **Core Classes**

- `ModelManager`: Model and tokenizer management
- `DocDataset`: Document generation and management
- `GCGOptimizer`: Hard prompt optimization
- `SoftPromptOptimizer`: Soft prompt optimization
- `TrainingVisualizer`: Results visualization
- `OptimizationConfig`: Configuration management

### **Utility Functions**

- `compute_dataset_kl`: KL divergence computation
- `plot_training_curves`: Legacy plotting function
- `load_model_tokenizer`: Legacy model loading

## 🔄 Migration Guide

### **From Original Code**

1. **Replace function calls with classes**:
   ```python
   # Old
   results, ids = optim_gcg(model, tokenizer, dataset, ...)
   
   # New
   optimizer = GCGOptimizer(model, tokenizer, dataset, config, log_fpath)
   results, ids = optimizer.optimize(n_epochs, kl_every)
   ```

2. **Use configuration objects**:
   ```python
   # Old
   batch_size=10, top_k=256, gamma=0.0
   
   # New
   config = OptimizationConfig(batch_size=10, top_k=256, gamma=0.0)
   ```

3. **Leverage new features**:
   ```python
   # Checkpointing
   optimizer.save_checkpoint("checkpoint.json")
   
   # Advanced visualization
   visualizer = TrainingVisualizer(results)
   visualizer.plot_training_curves("curves.png")
   ```

## 📄 License

This project is licensed under the same terms as the original codebase.

## 🙏 Acknowledgments

This refactoring maintains the original research contributions while making the code more maintainable and production-ready. The original paper and authors are acknowledged in the main README. 