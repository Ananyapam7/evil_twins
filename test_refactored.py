"""
Test script for the refactored evil_twins package.

This script demonstrates the new object-oriented API while maintaining
the same functionality as the original code.
"""

import logging
from evil_twins import (
    ModelManager,
    DocDataset,
    GCGOptimizer,
    SoftPromptOptimizer,
    OptimizationConfig,
    TrainingVisualizer,
    plot_training_curves,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "experiment_outputs"

def test_gcg_optimization():
    """Test GCG optimization with the new API."""
    logger.info("Testing GCG optimization...")
    
    # Load model and tokenizer using the new ModelManager
    model_manager = ModelManager("microsoft/phi-3-mini-4k-instruct", use_flash_attn_2=False)
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Tell me a recipe.",
        optim_prompt="! " * 5,
        n_docs=2,
        doc_len=8,
        gen_batch_size=1,
    )
    
    # Configure optimization
    config = OptimizationConfig(
        batch_size=1,
        top_k=10,
        gamma=0.0,
        early_stop_kl=0.0,
    )
    
    # Create and run optimizer
    optimizer = GCGOptimizer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        log_fpath="gcg_test_log.json",
        output_dir=OUTPUT_DIR,
    )
    
    # Run optimization
    results, best_prompt = optimizer.optimize(n_epochs=2, kl_every=1)
    
    logger.info(f"GCG optimization completed. Best prompt: {optimizer.get_optimized_prompt()}")
    
    # Create visualizations
    visualizer = TrainingVisualizer(results, output_dir=OUTPUT_DIR)
    visualizer.plot_training_curves("gcg_training_curves.png")
    visualizer.create_summary_report("gcg_summary.txt")
    
    return results, best_prompt


def test_soft_optimization():
    """Test soft prompt optimization with the new API."""
    logger.info("Testing soft prompt optimization...")
    
    # Load model and tokenizer
    model_manager = ModelManager("microsoft/phi-3-mini-4k-instruct", use_flash_attn_2=False)
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Tell me a recipe.",
        optim_prompt="! " * 3,
        n_docs=2,
        doc_len=8,
        gen_batch_size=1,
    )
    
    # Configure optimization
    config = OptimizationConfig(
        batch_size=1,
        learning_rate=1e-3,
    )
    
    # Create and run optimizer
    optimizer = SoftPromptOptimizer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        log_fpath="soft_test_log.json",
        emb_save_fpath="soft_test_embeddings.pt",
        output_dir=OUTPUT_DIR,
    )
    
    # Run optimization
    results, best_embeddings = optimizer.optimize(n_epochs=2, kl_every=1)
    
    logger.info(f"Soft optimization completed. Best embeddings shape: {best_embeddings.shape}")
    
    # Decode embeddings back to text
    decoded_text = optimizer.decode_embeddings(best_embeddings)
    logger.info(f"Decoded prompt: {decoded_text}")
    
    # Create visualizations
    visualizer = TrainingVisualizer(results, output_dir=OUTPUT_DIR)
    visualizer.plot_training_curves("soft_training_curves.png")
    visualizer.create_summary_report("soft_summary.txt")
    
    return results, best_embeddings


def test_legacy_compatibility():
    """Test that legacy functions still work."""
    logger.info("Testing legacy function compatibility...")
    
    from evil_twins import load_model_tokenizer, optim_gcg, optim_soft
    
    # Test legacy model loading
    model, tokenizer = load_model_tokenizer("microsoft/phi-3-mini-4k-instruct", use_flash_attn_2=False)
    
    # Test legacy GCG
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Tell me a recipe.",
        optim_prompt="! " * 3,
        n_docs=1,
        doc_len=4,
        gen_batch_size=1,
    )
    
    results, ids = optim_gcg(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        n_epochs=1,
        kl_every=1,
        log_fpath="legacy_gcg_log.json",
        batch_size=1,
        top_k=5,
        gamma=0.0,
        output_dir=OUTPUT_DIR,
    )
    
    logger.info("Legacy compatibility test passed!")
    return results, ids


def test_configuration():
    """Test configuration management."""
    logger.info("Testing configuration management...")
    
    # Test OptimizationConfig
    config = OptimizationConfig(
        batch_size=16,
        top_k=256,
        gamma=0.1,
        learning_rate=1e-3,
    )
    
    # Convert to dict and back
    config_dict = config.to_dict()
    config_from_dict = OptimizationConfig.from_dict(config_dict)
    
    assert config.batch_size == config_from_dict.batch_size
    assert config.top_k == config_from_dict.top_k
    assert config.gamma == config_from_dict.gamma
    assert config.learning_rate == config_from_dict.learning_rate
    
    logger.info("Configuration management test passed!")
    
    # Test ModelConfig
    from evil_twins import ModelConfig
    
    template = ModelConfig.get_template("microsoft/phi-3-mini-4k-instruct")
    assert template.prefix == ""
    assert template.suffix == ""
    
    logger.info("Model configuration test passed!")


def main():
    """Run all tests."""
    logger.info("Starting refactored evil_twins tests...")
    
    try:
        # Test configuration
        test_configuration()
        
        # Test GCG optimization
        gcg_results, gcg_prompt = test_gcg_optimization()
        
        # Test soft optimization
        soft_results, soft_embeddings = test_soft_optimization()
        
        # Test legacy compatibility
        legacy_results, legacy_ids = test_legacy_compatibility()
        
        logger.info("All tests completed successfully!")
        
        # Create comprehensive visualization
        from evil_twins import create_comprehensive_visualization
        create_comprehensive_visualization(f"{OUTPUT_DIR}/gcg_test_log.json", output_dir=OUTPUT_DIR)
        create_comprehensive_visualization(f"{OUTPUT_DIR}/soft_test_log.json", output_dir=OUTPUT_DIR)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 