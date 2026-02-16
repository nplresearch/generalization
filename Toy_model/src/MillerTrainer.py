"""
Training loop for the G-I tradeoff toy-model experiments.

MillerTrainer trains a model (TransformerModel or CompositionalModel) on
batches produced by a MillerDataGenerator / CompositionalDataGenerator.
The training objective combines two losses weighted by config-level lambdas:

  1. **Similarity loss** (NLL on softmax-normalised inner-product logits):
     encourages the model to rank the metrically closest context element
     highest.
  2. **Reconstruction loss** (MSE between the one-hot input and the
     reconstruction produced by the model's transpose embedding):
     acts as a regulariser that prevents the embedding from collapsing.

After each epoch the trainer evaluates G-score and I-score via
MillerTestEvaluator, logs all metrics in `train_history`, and adjusts
the learning rate with ReduceLROnPlateau (maximising G + I).

Usage (from notebooks)::

    trainer = MillerTrainer(model, data_gen, TrainerConfig(...))
    history = trainer.train(verbose=True)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MillerTest import MillerTestEvaluator

@dataclass
class TrainerConfig:
    """Configuration for Miller's Law training"""
    losses = ["reconstruction","similarity"] # List of losses to use
    num_epochs: int = 100 # Number of epochs to train
    learning_rate: float = 1e-3 # Learning rate for optimizer
    weight_decay: float = 1e-5 # Weight decay for optimizer
    patience: int = 10  # epochs before reducing learning rate
    min_lr: float = 1e-6 # Minimum learning rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MillerTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_generator: 'MillerDataGenerator',
        config: TrainerConfig
    ):
        # Initialize model, data generator, and configuration
        self.model = model.to(config.device)
        self.data_generator = data_generator
        self.config = config
        
        # Initialize optimizer and learning rate scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config.patience,
            min_lr=config.min_lr
        )
        
        
        # Initialize metric tracking for training history
        self.train_history = {
            'similarity_scores': [],
            'identification_scores': [],
            'similarity_losses': [],
            'reconstruction_losses': [],
            'identification_losses': [],
            'combined_losses': [],
            'confusion_matrix_sim': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch and return metrics"""
        self.model.train()  # Set model to training mode
        epoch_metrics = {
            'similarity_loss': 0.0,
            'identification_loss': 0.0,
            'reconstruction_loss': 0.0,
            'similarity_correct': 0,
            'identification_correct': 0,
            'total_samples': 0
        }
        # Iterate over batches
        loss_fn = nn.NLLLoss()
        loss_fn_rec = nn.MSELoss()
        for batch_idx in range(self.data_generator.config.num_samples // self.data_generator.config.batch_size):

            # Generate batches for similarity and identification tasks
            sim_indices, sim_labels = self.data_generator.generate_similarity_test_batch()
            
            sim_inputs = self.data_generator.indices_to_inputs(sim_indices)
            
            # Move data to the configured device 
            sim_inputs = sim_inputs.to(self.config.device)

            self.optimizer.zero_grad()

            # Forward pass for both tasks
            embs, recon = self.model(sim_inputs)
            ## Calculate losses for both tasks
            flag = 0
            if "similarity" in self.config.losses:
                if getattr(self.model, 'is_compositional', False):
                    slot_embs = self.model.get_slot_embeddings(sim_inputs)
                    per_slot_logits = []
                    for s in range(slot_embs.shape[2]):
                        ctx = slot_embs[:, :-1, s, :]
                        probe = slot_embs[:, [-1], s, :]
                        logits_s = torch.bmm(ctx, probe.permute(0, 2, 1))  # [B, K-1, 1]
                        per_slot_logits.append(logits_s)
                    sim_outputs = torch.max(torch.cat(per_slot_logits, dim=2), dim=2).values  # [B, K-1]
                    if isinstance(self.model.activation, torch.nn.ReLU):
                        sim_outputs = self.model.activation(sim_outputs) + 10**-10
                    else:
                        sim_outputs = self.model.activation(sim_outputs)
                else:
                    if isinstance(self.model.activation, torch.nn.ReLU):
                        sim_outputs = self.model.activation(torch.bmm(embs[:,:-1,:],embs[:,[-1],:].permute((0,2,1)) )) + 10**-10
                    else:
                        sim_outputs = self.model.activation(torch.bmm(embs[:,:-1,:],embs[:,[-1],:].permute((0,2,1)) ))# + 10**-10
                    sim_outputs = sim_outputs.squeeze(2)
                # Normalize by ratio (match evaluator): logits / sum(logits)
                sim_probs = sim_outputs/torch.sum(sim_outputs, 1, keepdims = True)
                # Similarity loss
                sim_loss = loss_fn(torch.log(sim_probs), sim_labels.to(self.config.device))
                combined_loss = self.config.lambda_sim*sim_loss
                flag = 1
                # Update metrics
                epoch_metrics['similarity_loss'] += self.config.lambda_sim*sim_loss.item()
                epoch_metrics['similarity_correct'] += ((sim_probs.cpu().gather(1,sim_labels.unsqueeze(1)))).sum().item()


            if "reconstruction" in self.config.losses:
                # Reconstruction loss
                rec_loss = loss_fn_rec(recon.reshape(-1,self.data_generator.config.num_inputs), sim_inputs.reshape(-1,self.data_generator.config.num_inputs))
                if flag == 1:
                    combined_loss += self.config.lambda_rec*rec_loss
                else:
                    combined_loss = self.config.lambda_rec*rec_loss
                # Update metrics
                epoch_metrics['reconstruction_loss'] += self.config.lambda_rec*rec_loss.item()

            # Backward pass and optimization step
            combined_loss.backward()

            self.optimizer.step()
  
            epoch_metrics['total_samples'] += sim_labels.size(0)
        
        # Compute final metrics for the epoch
        num_batches = batch_idx + 1
        metrics = {
            'similarity_loss': epoch_metrics['similarity_loss'] / num_batches,
            'similarity_accuracy': epoch_metrics['similarity_correct'] / epoch_metrics['total_samples'],
            'reconstruction_loss': epoch_metrics['reconstruction_loss'] / num_batches
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics"""
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            evaluator = MillerTestEvaluator(self.data_generator.config)
            g_score, i_score, confusion_matrix_sim = evaluator.evaluate_model(self.model, self.data_generator)
            
        return {
            'generalization_score': g_score,
            'identification_score': i_score,
            'confusion_matrix_sim': confusion_matrix_sim
        }
    
    def train(self, verbose: bool = True, printevery = 10, compute_embeddings = False) -> Dict[str, List[float]]:
        """Full training loop with validation"""
        best_combined_score = 0.0
        epochs_without_improvement = 0
        
        embs = []
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate scheduler based on validation score
            combined_score = val_metrics['generalization_score'] + val_metrics['identification_score']
            self.scheduler.step(combined_score)
 
            # Store metrics in training history
            self.train_history['similarity_scores'].append(val_metrics['generalization_score'])
            self.train_history['identification_scores'].append(val_metrics['identification_score'])
            self.train_history['similarity_losses'].append(train_metrics['similarity_loss'])
            self.train_history['reconstruction_losses'].append(train_metrics['reconstruction_loss'])
            self.train_history['confusion_matrix_sim'].append(val_metrics['confusion_matrix_sim'])
            if compute_embeddings:
                device = self.config.device
                num_inputs = getattr(self.data_generator.config, 'num_inputs', None)
                if num_inputs is None:
                    # Fallback: try to infer from model input dimension
                    num_inputs = getattr(self.config, 'num_inputs', None)
                x, __ = self.model(torch.eye(num_inputs).to(device).unsqueeze(0))
                x = x[0,:,:].detach().cpu().numpy()
                embs.append(x)
            
            # Check for improvement and save checkpoint if improved
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                epochs_without_improvement = 0
                #self._save_checkpoint(epoch, best_combined_score)
            else:
                epochs_without_improvement += 1
            
            # Print progress every 10 epochs if verbose
            if verbose and epoch % printevery == 0:
                print(f"Epoch {epoch}:")
                print(f"  Train - Sim Loss: {train_metrics['similarity_loss']:.4f}, "
                      f"Reconstruction Loss: {train_metrics['reconstruction_loss']:.4f}")
                print(f"  Val - G Score: {val_metrics['generalization_score']:.4f}, "
                      f"I Score: {val_metrics['identification_score']:.4f}")
        
        if compute_embeddings:
            return self.train_history, embs
        else:
            return self.train_history
    