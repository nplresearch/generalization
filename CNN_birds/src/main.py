import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Import necessary modules and classes
from MillerDataGen import MillerDataGenerator, DataGeneratorConfig, TaskType
from MillerDataVis import MillerVisualizer, VisualizationConfig
from MillerTest import MillerTestConfig, create_model
from MillerTrainer import MillerTrainer, TrainerConfig
from utils import ExperimentLogger, ConfigManager, ResultAggregator, create_default_config

class ExperimentRunner:
    """Manages and runs Miller's Law experiments"""
    def __init__(self):
        # Get project root directory (2 levels up from src)
        self.project_root = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        
        # Set up configurations
        self.setup_configs()
        
    def setup_directories(self):
        """Create experiment directories"""
        # Main project directories
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        
        # Experiment-specific directories
        self.exp_dir = self.results_dir / self.timestamp
        self.model_dir = self.data_dir / "models" / self.timestamp
        self.plots_dir = self.exp_dir / "plots"
        
        # Create all necessary directories
        for dir_path in [self.data_dir, self.results_dir, self.exp_dir, 
                        self.model_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_configs(self):
        """Initialize configurations for all components"""
        # Load base configuration
        config_manager = ConfigManager(self.project_root / "configs")
        try:
            config = config_manager.load_config("default")
        except FileNotFoundError:
            config = create_default_config()
            config_manager.save_config(config, "default")
        
        # Set component configurations from loaded config
        self.data_config = DataGeneratorConfig(
            num_inputs=config['data']['feature_dim'],
            sequence_len=config['data']['sequence_len'],
            batch_size=config['data']['batch_size'],
            noise_level=config['data']['noise_level'],
            num_samples=config['data'].get('num_samples', 1000)
        )
        
        self.model_config = MillerTestConfig(
            feature_dim=config['data']['feature_dim'],
            sequence_len=config['data']['sequence_len']
        )
        
        self.trainer_config = TrainerConfig(
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            checkpoint_dir=str(self.model_dir)
        )
        
        # Add visualization config
        self.vis_config = VisualizationConfig(
            figsize=tuple(config['visualization']['figsize']),  # Convert list to tuple
            dpi=config['visualization']['dpi'],
            style=config['visualization']['style'],
            save_dir=str(self.plots_dir),
            show_grid=True,
            confidence_intervals=True
        )

    def run_attention_comparison(self):
        """Compare different attention mechanisms"""
        attention_types = ["linear", "softmax"]
        histories = {}
        
        for att_type in attention_types:
            print(f"\nTraining model with {att_type} attention...")
            model = create_model(self.model_config, att_type)
            
            # Train on both linear and nonlinear tasks
            for task_type in [TaskType.LINEAR, TaskType.NONLINEAR]:
                self.data_config.task_type = task_type
                data_gen = MillerDataGenerator(self.data_config)
                
                trainer = MillerTrainer(model, data_gen, self.trainer_config)
                history = trainer.train(verbose=True)
                
                key = f"{att_type}_{task_type.value}"
                histories[key] = history
        
        return histories
    
    def run_task_comparison(self):
        """Compare performance on linear vs nonlinear tasks"""
        task_histories = {}
        model = create_model(self.model_config, "softmax")  # Use softmax attention as baseline
        
        # Test different difficulty levels
        difficulties = [0.2, 0.5, 0.8]
        
        for task_type in [TaskType.LINEAR, TaskType.NONLINEAR]:
            for diff in difficulties:
                print(f"\nTraining on {task_type.value} task with difficulty {diff}...")
                self.data_config.task_type = task_type
                self.data_config.difficulty = diff
                
                data_gen = MillerDataGenerator(self.data_config)
                trainer = MillerTrainer(model, data_gen, self.trainer_config)
                history = trainer.train(verbose=True)
                
                key = f"{task_type.value}_diff_{diff}"
                task_histories[key] = history
        
        return task_histories
    
    def run_systematic_evaluation(self):
        """Systematic evaluation of attention mechanisms"""
        # Test different hyperparameters
        learning_rates = [1e-4, 1e-3, 1e-2]
        noise_levels = [0.05, 0.1, 0.2]
        feature_dims = [32, 64, 128]
        
        results = {}
        
        for lr in learning_rates:
            for noise in noise_levels:
                for dim in feature_dims:
                    print(f"\nTesting LR={lr}, noise={noise}, dim={dim}")
                    
                    # Update configurations
                    self.trainer_config.learning_rate = lr
                    self.data_config.noise_level = noise
                    self.data_config.num_inputs = dim
                    self.model_config.feature_dim = dim
                    
                    # Test both attention types
                    for att_type in ["linear", "softmax"]:
                        model = create_model(self.model_config, att_type)
                        data_gen = MillerDataGenerator(self.data_config)
                        trainer = MillerTrainer(model, data_gen, self.trainer_config)
                        history = trainer.train(verbose=True)
                        
                        key = f"{att_type}_lr{lr}_noise{noise}_dim{dim}"
                        results[key] = history
        
        return results
    
    def visualize_results(self, histories: Dict[str, Dict], experiment_name: str):
        """Create visualizations for experiment results"""
        visualizer = MillerVisualizer(self.vis_config)
        
        # Plot comparison grid
        visualizer.plot_comparison_grid(histories, f"{experiment_name}_comparison")
        
        # Plot individual training summaries
        for model_name, history in histories.items():
            visualizer.create_training_summary(history, model_name)
            visualizer.plot_training_trajectory(history, model_name)
        
        # Plot Miller's curve for all models
        g_scores = []
        i_scores = []
        model_names = []
        
        for model_name, history in histories.items():
            g_scores.append(history['similarity_scores'][-1])
            i_scores.append(history['identification_scores'][-1])
            model_names.append(model_name)
        
        visualizer.plot_miller_curve(g_scores, i_scores, experiment_name)

def main():
    # Initialize experiment runner and utilities
    runner = ExperimentRunner()
    config_manager = ConfigManager(runner.project_root / "configs")
    logger = ExperimentLogger(runner.exp_dir, "miller_experiment")
    
    # Load or create configuration
    try:
        config = config_manager.load_config("default")
    except FileNotFoundError:
        config = create_default_config()
        config_manager.save_config(config, "default")
    
    # Log initial configuration
    logger.log_config(config)
    
    # Run experiments
    logger.logger.info("Starting attention mechanism comparison...")
    att_histories = runner.run_attention_comparison()
    runner.visualize_results(att_histories, "attention_comparison")
    
    logger.logger.info("Starting task comparison...")
    task_histories = runner.run_task_comparison()
    runner.visualize_results(task_histories, "task_comparison")
    
    logger.logger.info("Starting systematic evaluation...")
    eval_histories = runner.run_systematic_evaluation()
    runner.visualize_results(eval_histories, "systematic_evaluation")
    
    # Aggregate results
    aggregator = ResultAggregator(runner.results_dir)
    results_df = aggregator.aggregate_results()
    summary = aggregator.generate_summary_statistics(results_df)
    
    # Save summary
    aggregator.save_summary(summary, f"experiment_summary_{runner.timestamp}")
    logger.log_experiment_summary(summary)
    
    # Save final metrics
    metrics_path = runner.exp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'attention_comparison': att_histories,
            'task_comparison': task_histories,
            'systematic_evaluation': eval_histories,
            'summary': summary
        }, f, indent=2)

if __name__ == "__main__":
    main()