#!/usr/bin/env python3
"""
FRALA-ARC Training Script

This script trains the Fractal Reinforcement Learning Agent (FRALA) 
on Abstract Reasoning Corpus (ARC) tasks. It demonstrates how fractal
processing can solve visual reasoning problems requiring abstract pattern
recognition and few-shot learning.

Usage:
    python train_frala_arc.py --config configs/arc_config.json
    python train_frala_arc.py --synthetic --task-type pattern_completion
    python train_frala_arc.py --evaluate --model-path checkpoints/frala_arc.pt
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from datetime import datetime

from frala_arc import (
    FRALA_ARC, ARCTrainer, load_arc_task, visualize_grids
)


class SyntheticARCGenerator:
    """
    Generate synthetic ARC-like tasks for training and evaluation.
    Useful when real ARC dataset is not available.
    """
    
    def __init__(self, grid_size: int = 10, num_colors: int = 10):
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Define task types
        self.task_generators = {
            'copy_pattern': self._generate_copy_task,
            'color_mapping': self._generate_color_mapping_task,
            'pattern_completion': self._generate_pattern_completion_task,
            'symmetry': self._generate_symmetry_task,
            'scaling': self._generate_scaling_task,
            'rotation': self._generate_rotation_task,
            'object_counting': self._generate_counting_task
        }
    
    def generate_task(self, task_type: str, num_examples: int = 3) -> Dict[str, Any]:
        """Generate a synthetic ARC task of specified type."""
        
        if task_type not in self.task_generators:
            raise ValueError(f"Unknown task type: {task_type}")
        
        generator = self.task_generators[task_type]
        return generator(num_examples)
    
    def _generate_copy_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate copy and translate task."""
        
        train_examples = []
        
        for _ in range(num_examples):
            size = random.randint(3, 6)
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            output_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            # Create random pattern
            pattern = torch.randint(1, self.num_colors, (size, size))
            
            # Place original pattern
            start_x, start_y = random.randint(0, self.grid_size - size), random.randint(0, self.grid_size - size)
            input_grid[start_x:start_x+size, start_y:start_y+size] = pattern
            output_grid[start_x:start_x+size, start_y:start_y+size] = pattern
            
            # Place copied pattern
            offset_x, offset_y = random.randint(1, 3), random.randint(1, 3)
            copy_x = min(start_x + offset_x, self.grid_size - size)
            copy_y = min(start_y + offset_y, self.grid_size - size)
            output_grid[copy_x:copy_x+size, copy_y:copy_y+size] = pattern
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_size = random.randint(3, 6)
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_output = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        test_pattern = torch.randint(1, self.num_colors, (test_size, test_size))
        test_start_x, test_start_y = random.randint(0, self.grid_size - test_size), random.randint(0, self.grid_size - test_size)
        test_input[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_pattern
        test_output[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_pattern
        
        test_offset_x, test_offset_y = random.randint(1, 3), random.randint(1, 3)
        test_copy_x = min(test_start_x + test_offset_x, self.grid_size - test_size)
        test_copy_y = min(test_start_y + test_offset_y, self.grid_size - test_size)
        test_output[test_copy_x:test_copy_x+test_size, test_copy_y:test_copy_y+test_size] = test_pattern
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_color_mapping_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate color mapping task (replace color A with color B)."""
        
        # Define the color mapping
        source_color = random.randint(1, self.num_colors - 1)
        target_color = random.randint(1, self.num_colors - 1)
        while target_color == source_color:
            target_color = random.randint(1, self.num_colors - 1)
        
        train_examples = []
        
        for _ in range(num_examples):
            # Create random pattern with source color
            input_grid = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
            output_grid = input_grid.clone()
            
            # Apply color mapping
            output_grid[input_grid == source_color] = target_color
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_input = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
        test_output = test_input.clone()
        test_output[test_input == source_color] = target_color
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_pattern_completion_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate pattern completion task."""
        
        train_examples = []
        
        for _ in range(num_examples):
            # Create a pattern with missing piece
            size = random.randint(4, 7)
            full_pattern = torch.randint(1, self.num_colors, (size, size))
            
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            output_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            start_x, start_y = random.randint(0, self.grid_size - size), random.randint(0, self.grid_size - size)
            
            # Place full pattern in output
            output_grid[start_x:start_x+size, start_y:start_y+size] = full_pattern
            
            # Create incomplete pattern in input (remove some pieces)
            incomplete_pattern = full_pattern.clone()
            mask_size = random.randint(1, size // 2)
            mask_x, mask_y = random.randint(0, size - mask_size), random.randint(0, size - mask_size)
            incomplete_pattern[mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = 0
            
            input_grid[start_x:start_x+size, start_y:start_y+size] = incomplete_pattern
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_size = random.randint(4, 7)
        test_full_pattern = torch.randint(1, self.num_colors, (test_size, test_size))
        
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_output = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        test_start_x, test_start_y = random.randint(0, self.grid_size - test_size), random.randint(0, self.grid_size - test_size)
        test_output[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_full_pattern
        
        test_incomplete = test_full_pattern.clone()
        test_mask_size = random.randint(1, test_size // 2)
        test_mask_x, test_mask_y = random.randint(0, test_size - test_mask_size), random.randint(0, test_size - test_mask_size)
        test_incomplete[test_mask_x:test_mask_x+test_mask_size, test_mask_y:test_mask_y+test_mask_size] = 0
        
        test_input[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_incomplete
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_symmetry_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate symmetry completion task."""
        
        train_examples = []
        
        for _ in range(num_examples):
            # Create asymmetric pattern
            half_size = random.randint(3, 5)
            pattern = torch.randint(1, self.num_colors, (self.grid_size, half_size))
            
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            output_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            # Place original pattern
            input_grid[:, :half_size] = pattern
            output_grid[:, :half_size] = pattern
            
            # Create symmetric pattern
            output_grid[:, half_size:2*half_size] = torch.flip(pattern, dims=[1])
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_half_size = random.randint(3, 5)
        test_pattern = torch.randint(1, self.num_colors, (self.grid_size, test_half_size))
        
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_output = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        test_input[:, :test_half_size] = test_pattern
        test_output[:, :test_half_size] = test_pattern
        test_output[:, test_half_size:2*test_half_size] = torch.flip(test_pattern, dims=[1])
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_scaling_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate scaling task (make pattern larger/smaller)."""
        
        train_examples = []
        scale_factor = 2
        
        for _ in range(num_examples):
            # Create small pattern
            small_size = random.randint(2, 4)
            small_pattern = torch.randint(1, self.num_colors, (small_size, small_size))
            
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            output_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            # Place small pattern
            start_x, start_y = random.randint(0, self.grid_size - small_size), random.randint(0, self.grid_size - small_size)
            input_grid[start_x:start_x+small_size, start_y:start_y+small_size] = small_pattern
            
            # Create scaled pattern
            large_size = small_size * scale_factor
            if large_size <= self.grid_size:
                large_pattern = small_pattern.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
                large_start_x = min(start_x, self.grid_size - large_size)
                large_start_y = min(start_y, self.grid_size - large_size)
                output_grid[large_start_x:large_start_x+large_size, large_start_y:large_start_y+large_size] = large_pattern
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case  
        test_small_size = random.randint(2, 4)
        test_small_pattern = torch.randint(1, self.num_colors, (test_small_size, test_small_size))
        
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_output = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        test_start_x, test_start_y = random.randint(0, self.grid_size - test_small_size), random.randint(0, self.grid_size - test_small_size)
        test_input[test_start_x:test_start_x+test_small_size, test_start_y:test_start_y+test_small_size] = test_small_pattern
        
        test_large_size = test_small_size * scale_factor
        if test_large_size <= self.grid_size:
            test_large_pattern = test_small_pattern.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
            test_large_start_x = min(test_start_x, self.grid_size - test_large_size)
            test_large_start_y = min(test_start_y, self.grid_size - test_large_size)
            test_output[test_large_start_x:test_large_start_x+test_large_size, test_large_start_y:test_large_start_y+test_large_size] = test_large_pattern
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_rotation_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate rotation task."""
        
        train_examples = []
        
        for _ in range(num_examples):
            # Create asymmetric pattern
            size = random.randint(3, 6)
            pattern = torch.zeros(size, size, dtype=torch.long)
            
            # Create L-shaped pattern
            pattern[:size//2, :] = random.randint(1, self.num_colors - 1)
            pattern[size//2:, :size//2] = random.randint(1, self.num_colors - 1)
            
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            output_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            # Place original pattern
            start_x, start_y = random.randint(0, self.grid_size - size), random.randint(0, self.grid_size - size)
            input_grid[start_x:start_x+size, start_y:start_y+size] = pattern
            
            # Rotate pattern 90 degrees
            rotated = torch.rot90(pattern, k=1, dims=[0, 1])
            output_grid[start_x:start_x+size, start_y:start_y+size] = rotated
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_size = random.randint(3, 6)
        test_pattern = torch.zeros(test_size, test_size, dtype=torch.long)
        test_pattern[:test_size//2, :] = random.randint(1, self.num_colors - 1)
        test_pattern[test_size//2:, :test_size//2] = random.randint(1, self.num_colors - 1)
        
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_output = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        test_start_x, test_start_y = random.randint(0, self.grid_size - test_size), random.randint(0, self.grid_size - test_size)
        test_input[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_pattern
        
        test_rotated = torch.rot90(test_pattern, k=1, dims=[0, 1])
        test_output[test_start_x:test_start_x+test_size, test_start_y:test_start_y+test_size] = test_rotated
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }
    
    def _generate_counting_task(self, num_examples: int) -> Dict[str, Any]:
        """Generate object counting task."""
        
        train_examples = []
        target_color = random.randint(1, self.num_colors - 1)
        
        for _ in range(num_examples):
            input_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
            
            # Place random colored objects
            num_objects = random.randint(1, 5)
            for _ in range(num_objects):
                obj_size = random.randint(1, 3)
                obj_x, obj_y = random.randint(0, self.grid_size - obj_size), random.randint(0, self.grid_size - obj_size)
                obj_color = random.randint(1, self.num_colors - 1)
                input_grid[obj_x:obj_x+obj_size, obj_y:obj_y+obj_size] = obj_color
            
            # Count target color objects
            count = (input_grid == target_color).sum().item()
            
            # Output grid shows count in top-left corner
            output_grid = input_grid.clone()
            if count > 0 and count <= 9:
                output_grid[0, 0] = count
            
            train_examples.append({
                'input': input_grid.numpy().tolist(),
                'output': output_grid.numpy().tolist()
            })
        
        # Generate test case
        test_input = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        test_num_objects = random.randint(1, 5)
        
        for _ in range(test_num_objects):
            test_obj_size = random.randint(1, 3)
            test_obj_x, test_obj_y = random.randint(0, self.grid_size - test_obj_size), random.randint(0, self.grid_size - test_obj_size)
            test_obj_color = random.randint(1, self.num_colors - 1)
            test_input[test_obj_x:test_obj_x+test_obj_size, test_obj_y:test_obj_y+test_obj_size] = test_obj_color
        
        test_count = (test_input == target_color).sum().item()
        test_output = test_input.clone()
        if test_count > 0 and test_count <= 9:
            test_output[0, 0] = test_count
        
        return {
            'train': {
                'input': [ex['input'] for ex in train_examples],
                'output': [ex['output'] for ex in train_examples]
            },
            'test': [{
                'input': test_input.numpy().tolist(),
                'output': test_output.numpy().tolist()
            }]
        }


class ARCExperiment:
    """
    Comprehensive experiment framework for FRALA-ARC training and evaluation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        exp_name: str = "frala_arc",
        save_dir: str = "./arc_experiments"
    ):
        self.config = config
        self.exp_name = exp_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create model
        self.model = FRALA_ARC(**config['model'])
        
        # Create trainer
        self.trainer = ARCTrainer(
            self.model,
            learning_rate=config['training']['learning_rate']
        )
        
        # Synthetic task generator (if needed)
        if config.get('use_synthetic', False) or config['training'].get('use_synthetic', False):
            self.synthetic_generator = SyntheticARCGenerator(
                grid_size=config['model']['grid_size'],
                num_colors=config['model']['num_colors']
            )
        
        # Tracking
        self.training_history = []
        self.evaluation_results = []
        
        print(f"🧩 FRALA-ARC Experiment: {exp_name}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"💾 Save directory: {save_dir}")
    
    def train_synthetic(
        self,
        task_types: List[str],
        num_tasks_per_type: int = 50,
        num_epochs_per_task: int = 100
    ):
        """Train on synthetic tasks."""
        
        print(f"\n🏋️ Training on Synthetic Tasks")
        print(f"Task types: {task_types}")
        print(f"Tasks per type: {num_tasks_per_type}")
        print(f"Epochs per task: {num_epochs_per_task}")
        
        all_losses = []
        
        for task_type in task_types:
            print(f"\n--- Training on {task_type} tasks ---")
            type_losses = []
            
            for task_idx in tqdm(range(num_tasks_per_type), desc=f"{task_type}"):
                # Generate synthetic task
                task = self.synthetic_generator.generate_task(task_type)
                
                # Train on this task
                loss = self.trainer.train_on_task(task, num_epochs_per_task)
                type_losses.append(loss)
                
                # Log progress
                if task_idx % 10 == 0:
                    avg_loss = np.mean(type_losses[-10:])
                    print(f"  Task {task_idx:3d}, Avg Loss: {avg_loss:.4f}")
            
            avg_type_loss = np.mean(type_losses)
            print(f"{task_type} average loss: {avg_type_loss:.4f}")
            
            all_losses.extend(type_losses)
            
            # Save checkpoint
            self._save_checkpoint(f"after_{task_type}")
        
        self.training_history.extend(all_losses)
        print(f"\n✅ Synthetic training complete. Overall avg loss: {np.mean(all_losses):.4f}")
    
    def evaluate_synthetic(
        self,
        task_types: List[str],
        num_eval_tasks: int = 20
    ) -> Dict[str, float]:
        """Evaluate on synthetic tasks."""
        
        print(f"\n🔍 Evaluating on Synthetic Tasks")
        results = {}
        
        for task_type in task_types:
            print(f"\n--- Evaluating {task_type} ---")
            accuracies = []
            
            for _ in range(num_eval_tasks):
                # Generate evaluation task
                task = self.synthetic_generator.generate_task(task_type)
                
                # Evaluate
                eval_result = self.trainer.evaluate_task(task)
                accuracies.append(eval_result['accuracy'])
            
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            results[task_type] = {
                'accuracy': avg_accuracy,
                'std': std_accuracy,
                'all_accuracies': accuracies
            }
            
            print(f"{task_type}: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
        
        self.evaluation_results.append({
            'type': 'synthetic',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
        return results
    
    def train_on_arc_dataset(
        self,
        dataset_path: str,
        num_tasks: Optional[int] = None,
        num_epochs_per_task: int = 200
    ):
        """Train on real ARC dataset."""
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"ARC dataset not found: {dataset_path}")
        
        # Load ARC tasks
        task_files = list(dataset_path.glob("*.json"))
        if num_tasks:
            task_files = task_files[:num_tasks]
        
        print(f"\n🏋️ Training on ARC Dataset")
        print(f"Dataset path: {dataset_path}")
        print(f"Number of tasks: {len(task_files)}")
        print(f"Epochs per task: {num_epochs_per_task}")
        
        losses = []
        
        for i, task_file in enumerate(tqdm(task_files, desc="Training")):
            # Load task
            task = load_arc_task(str(task_file))
            
            # Train on this task
            loss = self.trainer.train_on_task(task, num_epochs_per_task)
            losses.append(loss)
            
            # Log progress
            if i % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                print(f"Task {i:3d}, Avg Loss: {avg_loss:.4f}")
            
            # Save periodic checkpoints
            if i % 50 == 0:
                self._save_checkpoint(f"arc_task_{i}")
        
        self.training_history.extend(losses)
        print(f"\n✅ ARC training complete. Average loss: {np.mean(losses):.4f}")
    
    def evaluate_arc_dataset(
        self,
        dataset_path: str,
        num_tasks: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate on real ARC dataset."""
        
        dataset_path = Path(dataset_path)
        task_files = list(dataset_path.glob("*.json"))
        if num_tasks:
            task_files = task_files[:num_tasks]
        
        print(f"\n🔍 Evaluating on ARC Dataset")
        print(f"Number of tasks: {len(task_files)}")
        
        accuracies = []
        detailed_results = []
        
        for task_file in tqdm(task_files, desc="Evaluating"):
            # Load and evaluate task
            task = load_arc_task(str(task_file))
            result = self.trainer.evaluate_task(task)
            
            accuracies.append(result['accuracy'])
            detailed_results.append({
                'task_file': task_file.name,
                'accuracy': result['accuracy'],
                'analysis': result.get('analysis', {})
            })
        
        overall_results = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'num_tasks': len(task_files),
            'detailed_results': detailed_results
        }
        
        self.evaluation_results.append({
            'type': 'arc_dataset',
            'results': overall_results,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Average accuracy: {overall_results['avg_accuracy']:.3f} ± {overall_results['std_accuracy']:.3f}")
        
        return overall_results
    
    def visualize_predictions(
        self,
        task_data: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Visualize model predictions on a task."""
        
        # Get model prediction
        result = self.trainer.evaluate_task(task_data)
        
        # Prepare grids for visualization
        grids = []
        titles = []
        
        # Training examples
        for i, (inp, out) in enumerate(zip(task_data['train']['input'], task_data['train']['output'])):
            grids.extend([np.array(inp), np.array(out)])
            titles.extend([f"Train {i+1} Input", f"Train {i+1} Output"])
        
        # Test example
        test_input = np.array(task_data['test'][0]['input'])
        test_target = np.array(task_data['test'][0]['output'])
        test_prediction = result['predicted_grid']
        
        grids.extend([test_input, test_target, test_prediction])
        titles.extend(["Test Input", "Test Target", "Test Prediction"])
        
        # Create visualization
        visualize_grids(grids, titles)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return result
    
    def analyze_fractal_processing(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how fractal processing works on a specific task."""
        
        print("\n🔬 Fractal Processing Analysis")
        
        # Analyze the task
        input_grids = [torch.tensor(grid) for grid in task_data['train']['input']]
        output_grids = [torch.tensor(grid) for grid in task_data['train']['output']]
        
        analysis = self.model.analyze_task(input_grids, output_grids)
        
        print(f"📊 Task Overview:")
        print(f"  Examples: {analysis['num_examples']}")
        print(f"  Grid shapes: {analysis['grid_shapes']}")
        
        print(f"\n🎨 Color Usage:")
        for i, color_info in enumerate(analysis['color_usage']):
            print(f"  Example {i+1}: {color_info['input_colors']} → {color_info['output_colors']}")
            if color_info['color_change']:
                print(f"    New colors: {color_info['color_change']}")
        
        print(f"\n🌀 Fractal Analysis:")
        for i, fractal_info in enumerate(analysis['fractal_analysis']):
            print(f"  Example {i+1}:")
            print(f"    Level norms: {[f'{norm:.2f}' for norm in fractal_info['level_feature_norms']]}")
            
            # Show top rules by confidence
            confidences = fractal_info['rule_confidences']
            sorted_rules = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
            top_rules = sorted_rules[:3]
            
            print(f"    Top rules: {[(rule, f'{conf:.3f}') for rule, conf in top_rules]}")
        
        return analysis
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint and training state."""
        
        checkpoint_dir = self.save_dir / f"{self.exp_name}_{checkpoint_name}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }, checkpoint_dir / "model.pt")
        
        # Save training history
        with open(checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation results
        with open(checkpoint_dir / "evaluation_results.json", 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    def save_final_results(self):
        """Save final experimental results."""
        
        final_dir = self.save_dir / f"{self.exp_name}_final"
        final_dir.mkdir(exist_ok=True)
        
        # Save complete results
        results = {
            'experiment_name': self.exp_name,
            'config': self.config,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(final_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, final_dir / "final_model.pt")
        
        print(f"🎯 Final results saved: {final_dir}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for FRALA-ARC."""
    
    return {
        'model': {
            'grid_size': 30,
            'num_colors': 10,
            'embed_dim': 256,
            'num_fractal_levels': 4,
            'heads': 8
        },
        'training': {
            'learning_rate': 0.001,
            'num_epochs_per_task': 100,
            'use_synthetic': True
        },
        'evaluation': {
            'num_eval_tasks': 20
        },
        'synthetic_tasks': [
            'copy_pattern',
            'color_mapping',
            'pattern_completion',
            'symmetry',
            'scaling'
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Train FRALA-ARC on ARC tasks")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic tasks")
    parser.add_argument("--task-type", type=str, default="copy_pattern", 
                       help="Synthetic task type")
    parser.add_argument("--arc-dataset", type=str, help="Path to ARC dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluation mode")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--exp-name", type=str, default="frala_arc", 
                       help="Experiment name")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        config['training']['use_synthetic'] = args.synthetic
    
    # Create experiment
    experiment = ARCExperiment(config, args.exp_name)
    
    if args.evaluate:
        # Evaluation mode
        if args.model_path:
            # Load trained model
            checkpoint = torch.load(args.model_path)
            experiment.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📁 Loaded model from: {args.model_path}")
        
        if args.arc_dataset:
            # Evaluate on ARC dataset
            results = experiment.evaluate_arc_dataset(args.arc_dataset)
            print(f"🎯 ARC Dataset Results: {results['avg_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        
        if config['training']['use_synthetic']:
            # Evaluate on synthetic tasks
            results = experiment.evaluate_synthetic(config['synthetic_tasks'])
            for task_type, result in results.items():
                print(f"🎯 {task_type}: {result['accuracy']:.3f} ± {result['std']:.3f}")
    
    else:
        # Training mode
        if config['training']['use_synthetic']:
            # Train on synthetic tasks
            experiment.train_synthetic(
                config['synthetic_tasks'],
                num_tasks_per_type=50,
                num_epochs_per_task=config['training']['num_epochs_per_task']
            )
            
            # Evaluate synthetic performance
            experiment.evaluate_synthetic(config['synthetic_tasks'])
        
        if args.arc_dataset:
            # Train on real ARC dataset
            experiment.train_on_arc_dataset(
                args.arc_dataset,
                num_epochs_per_task=config['training']['num_epochs_per_task']
            )
            
            # Evaluate ARC performance
            experiment.evaluate_arc_dataset(args.arc_dataset)
    
    # Save final results
    experiment.save_final_results()
    
    print("\n🎉 FRALA-ARC experiment completed!")


if __name__ == "__main__":
    main() 