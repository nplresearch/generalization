import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    style: str = 'seaborn'
    cmap: str = 'viridis'
    save_dir: str = './figures'
    show_grid: bool = True
    confidence_intervals: bool = True

class MillerVisualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        sns.set(style=self.config.style)
        
        # Ensure save directory exists
        os.makedirs(self.config.save_dir, exist_ok=True)

    def plot_miller_curve(self, 
                         g_scores: List[float], 
                         i_scores: List[float],
                         model_name: str,
                         show_theoretical: bool = True) -> None:
        """Plot experimental results on Miller's curve"""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot experimental points
        plt.scatter(g_scores, i_scores, label=f'{model_name} (experimental)',
                   alpha=0.6, s=50)
        
        # Plot theoretical curve if requested
        if show_theoretical:
            x = np.linspace(0.5, 1.0, 100)
            y = 1 - (x - 0.5)  # Theoretical relationship
            plt.plot(x, y, '--', label='Theoretical bound', color='red', alpha=0.7)
        
        plt.xlabel('Generalization Score (G)')
        plt.ylabel('Identification Score (I)')
        plt.title("Miller's Law Trade-off Curve")
        plt.grid(self.config.show_grid)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.config.save_dir, f"miller_curve_{model_name}.png"))
        plt.close()

    def plot_training_trajectory(self,
                               history: Dict[str, List[float]],
                               model_name: str) -> None:
        """Plot training trajectory on Miller's curve"""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        g_scores = history['similarity_scores']
        i_scores = history['identification_scores']
        
        # Create color gradient for trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, len(g_scores)))
        
        # Plot trajectory with color gradient
        for i in range(len(g_scores)-1):
            plt.plot([g_scores[i], g_scores[i+1]], 
                    [i_scores[i], i_scores[i+1]], 
                    color=colors[i], alpha=0.6)
        
        # Plot points
        scatter = plt.scatter(g_scores, i_scores, 
                            c=np.arange(len(g_scores)),
                            cmap=self.config.cmap,
                            label=model_name)
        
        # Add colorbar
        plt.colorbar(scatter, label='Training Epoch')
        
        plt.xlabel('Generalization Score (G)')
        plt.ylabel('Identification Score (I)')
        plt.title(f"Training Trajectory on Miller's Curve - {model_name}")
        plt.grid(self.config.show_grid)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.config.save_dir, f"training_trajectory_{model_name}.png"))
        plt.close()

    def plot_comparison_grid(self, histories: Dict[str, Dict], filename: str) -> None:
        """Create grid of comparison plots for different models"""
        metrics = ['similarity_scores', 'identification_scores', 
                  'similarity_losses', 'identification_losses']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), 
                                dpi=self.config.dpi)
        fig.suptitle('Model Comparison Grid')
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            for model_name, history in histories.items():
                if metric in history:  # Check if metric exists in history
                    axes[row, col].plot(history[metric], 
                                      label=model_name, 
                                      alpha=0.7)
            
            axes[row, col].set_title(metric.replace('_', ' ').title())
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(self.config.show_grid)
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, f"{filename}.png"))
        plt.close()

    def create_training_summary(self,
                              history: Dict[str, List[float]],
                              model_name: str) -> None:
        """Create comprehensive training summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.config.dpi)
        fig.suptitle(f'Training Summary - {model_name}')
        
        # Plot losses
        if 'similarity_losses' in history and 'identification_losses' in history:
            axes[0, 0].plot(history['similarity_losses'], label='Similarity')
            axes[0, 0].plot(history['identification_losses'], label='Identification')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(self.config.show_grid)
        
        # Plot accuracies
        if 'similarity_scores' in history and 'identification_scores' in history:
            axes[0, 1].plot(history['similarity_scores'], label='Similarity')
            axes[0, 1].plot(history['identification_scores'], label='Identification')
            axes[0, 1].set_title('Task Accuracies')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(self.config.show_grid)
        
        # Plot trade-off trajectory
        if 'similarity_scores' in history and 'identification_scores' in history:
            scatter = axes[1, 0].scatter(history['similarity_scores'],
                                       history['identification_scores'],
                                       c=np.arange(len(history['similarity_scores'])),
                                       cmap=self.config.cmap)
            axes[1, 0].set_title("Miller's Trade-off Trajectory")
            axes[1, 0].set_xlabel('Generalization Score (G)')
            axes[1, 0].set_ylabel('Identification Score (I)')
            axes[1, 0].grid(self.config.show_grid)
            plt.colorbar(scatter, ax=axes[1, 0], label='Epoch')
        
        # Plot combined performance
        if 'similarity_scores' in history and 'identification_scores' in history:
            combined_score = [g + i for g, i in 
                            zip(history['similarity_scores'],
                                history['identification_scores'])]
            axes[1, 1].plot(combined_score, label='Combined Performance')
            axes[1, 1].set_title('Combined Performance')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('G + I Score')
            axes[1, 1].grid(self.config.show_grid)
        
        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
        plt.savefig(os.path.join(self.config.save_dir, f"{model_name}_summary.png"))
        plt.close()