import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class ExperimentLogger:
    """Handles logging for Miller's Law experiments"""
    def __init__(self, exp_dir: Path, experiment_name: str):
        self.exp_dir = exp_dir
        self.log_dir = exp_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up file and console logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / f"{experiment_name}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a given step"""
        self.logger.info(f"Step {step} metrics:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
    
    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log experiment summary"""
        self.logger.info("Experiment Summary:")
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")

class ConfigManager:
    """Manages configuration for Miller's Law experiments"""
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], name: str):
        """Save configuration to YAML file"""
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_experiment_config(self, base_config: Dict[str, Any], 
                               overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Create experiment config with overrides"""
        config = base_config.copy()
        config.update(overrides)
        return config

class ResultAggregator:
    """Aggregates and analyzes results across multiple experiments"""
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
    
    def load_experiment_results(self, exp_dir: Path) -> Dict[str, Any]:
        """Load results from a single experiment"""
        metrics_path = exp_dir / "metrics.json"
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: No metrics file found in {exp_dir}")
            return None
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {exp_dir}/metrics.json")
            return None
    
    def aggregate_results(self) -> pd.DataFrame:
        """Aggregate results across all experiments"""
        results = []
        
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            exp_results = self.load_experiment_results(exp_dir)
            if exp_results is not None:
                # Extract relevant metrics
                for model_type, history in exp_results.get('attention_comparison', {}).items():
                    if isinstance(history, dict):
                        result = {
                            'timestamp': exp_dir.name,
                            'model_type': model_type,
                            'similarity_score': history.get('similarity_scores', [])[-1] if history.get('similarity_scores') else None,
                            'identification_score': history.get('identification_scores', [])[-1] if history.get('identification_scores') else None
                        }
                        results.append(result)
        
        if not results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'model_type', 'similarity_score', 'identification_score'])
        
        return pd.DataFrame(results)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for aggregated results"""
        if df.empty:
            return {
                'num_experiments': 0,
                'avg_similarity': None,
                'avg_identification': None,
                'best_combined_score': None,
                'timestamp_range': None
            }
            
        summary = {
            'num_experiments': len(df),
            'avg_similarity': df['similarity_score'].mean() if 'similarity_score' in df else None,
            'avg_identification': df['identification_score'].mean() if 'identification_score' in df else None,
            'best_combined_score': (df['similarity_score'] + df['identification_score']).max() if 'similarity_score' in df and 'identification_score' in df else None,
            'timestamp_range': [df['timestamp'].min(), df['timestamp'].max()] if 'timestamp' in df else None
        }
        
        # Group by model type if we have any results
        if not df.empty and 'model_type' in df:
            model_stats = df.groupby('model_type').agg({
                'similarity_score': ['mean', 'std'],
                'identification_score': ['mean', 'std']
            }).to_dict()
            
            summary['model_type_stats'] = model_stats
            
        return summary
        
    def save_summary(self, summary: Dict[str, Any], name: str):
        """Save summary to JSON file"""
        try:
            # Create summaries directory if it doesn't exist
            summary_dir = self.results_dir / "summaries"
            summary_dir.mkdir(exist_ok=True)
            
            # Save summary
            summary_path = summary_dir / f"{name}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
                
            print(f"Summary saved to {summary_path}")
            
        except Exception as e:
            print(f"Error saving summary: {e}")

# Example configuration file structure (config.yaml):
"""
data:
  feature_dim: 64
  sequence_len: 3
  batch_size: 32
  noise_level: 0.1
  task_type: linear

model:
  attention_type: softmax
  hidden_dim: 128
  num_layers: 2

training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  patience: 10

visualization:
  figsize: [10, 8]
  dpi: 100
  style: seaborn
"""

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'data': {
            'feature_dim': 64,
            'sequence_len': 3,
            'batch_size': 32,
            'noise_level': 0.1,
            'task_type': 'linear',
            'num_samples': 1000  # Added num_samples
        },
        'model': {
            'attention_type': 'softmax',
            'hidden_dim': 128,
            'num_layers': 2
        },
        'training': {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'patience': 10
        },
        'visualization': {
            'figsize': [10, 8],
            'dpi': 100,
            'style': 'seaborn'
        }
    }