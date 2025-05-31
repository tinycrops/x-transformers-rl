#!/usr/bin/env python3
"""
Train FRALA-ARC on Full ARC Training Dataset

This script trains on the complete ARC training dataset (400 tasks) with 
optimizations for large-scale training including curriculum learning,
adaptive learning rates, and progressive training strategies.
"""

import torch
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from frala_arc import FRALA_ARC, ARCTrainer
import os
import time

def analyze_task_difficulty(task_data):
    """Analyze task difficulty for curriculum learning."""
    difficulty_score = 0
    
    # Check number of training examples (fewer = harder)
    num_examples = len(task_data['train'])
    difficulty_score += max(0, 5 - num_examples) * 0.2
    
    # Check grid sizes (larger = harder)
    max_size = 0
    for example in task_data['train']:
        input_h, input_w = len(example['input']), len(example['input'][0])
        output_h, output_w = len(example['output']), len(example['output'][0])
        max_size = max(max_size, input_h, input_w, output_h, output_w)
    
    difficulty_score += (max_size / 30.0) * 0.3
    
    # Check color complexity (more colors = harder)
    all_colors = set()
    for example in task_data['train']:
        for row in example['input']:
            all_colors.update(row)
        for row in example['output']:
            all_colors.update(row)
    
    difficulty_score += (len(all_colors) / 10.0) * 0.2
    
    # Check output shape consistency (inconsistent = harder)
    output_shapes = []
    for example in task_data['train']:
        shape = (len(example['output']), len(example['output'][0]))
        output_shapes.append(shape)
    
    if len(set(output_shapes)) > 1:
        difficulty_score += 0.3
    
    return min(difficulty_score, 1.0)

def load_arc_training_tasks_full(arc_data_dir: str, max_tasks: int = None, curriculum: bool = True):
    """Load and optionally sort ARC training tasks by difficulty."""
    training_dir = Path(arc_data_dir) / "training"
    task_files = list(training_dir.glob("*.json"))
    
    print(f"📚 Loading ALL {len(task_files)} ARC training tasks...")
    
    tasks = []
    for task_file in tqdm(task_files, desc="Loading tasks"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            
            # Validate task structure
            if 'train' in task_data and len(task_data['train']) > 0:
                # Check grid sizes (allow larger grids but cap at 30)
                max_size = 30  # Model limit
                valid_task = True
                
                for example in task_data['train']:
                    input_h, input_w = len(example['input']), len(example['input'][0])
                    output_h, output_w = len(example['output']), len(example['output'][0])
                    
                    if input_h > max_size or input_w > max_size or output_h > max_size or output_w > max_size:
                        valid_task = False
                        break
                
                if valid_task:
                    difficulty = analyze_task_difficulty(task_data) if curriculum else 0.5
                    tasks.append({
                        'task_id': task_file.stem,
                        'data': task_data,
                        'difficulty': difficulty
                    })
        
        except Exception as e:
            print(f"❌ Error loading {task_file}: {e}")
            continue
    
    if curriculum:
        # Sort by difficulty (easy to hard)
        tasks.sort(key=lambda x: x['difficulty'])
        print(f"📊 Tasks sorted by difficulty (curriculum learning)")
        print(f"   Easiest task difficulty: {tasks[0]['difficulty']:.3f}")
        print(f"   Hardest task difficulty: {tasks[-1]['difficulty']:.3f}")
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    print(f"✅ Successfully loaded {len(tasks)} valid ARC tasks")
    return tasks

def train_on_full_arc(
    model_path: str = None,
    arc_data_dir: str = "arc_dataset/data",
    max_tasks: int = 400,  # Use all available tasks
    epochs_per_task: int = 40,  # Balanced epochs
    batch_tasks: int = 8,  # Smaller batches for memory efficiency
    exp_name: str = "full_arc_training",
    curriculum: bool = True,
    adaptive_lr: bool = True,
    save_frequency: int = 5  # Save every 5 batches
):
    """Train FRALA-ARC on full ARC training data with optimizations."""
    
    print("🚀 FRALA-ARC Full Dataset Training")
    print("=" * 60)
    print(f"📊 Target tasks: {max_tasks}")
    print(f"📚 Curriculum learning: {curriculum}")
    print(f"🎯 Adaptive learning rate: {adaptive_lr}")
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"📁 Loading existing model from {model_path}")
        model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("🆕 Creating new model")
        model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    
    # Start with lower learning rate for large dataset
    initial_lr = 0.0003 if adaptive_lr else 0.0005
    trainer = ARCTrainer(model, learning_rate=initial_lr)
    
    # Load real ARC tasks
    tasks = load_arc_training_tasks_full(arc_data_dir, max_tasks, curriculum)
    
    if not tasks:
        print("❌ No valid tasks found!")
        return
    
    # Create experiment directory
    exp_dir = Path("arc_experiments") / exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    # Training setup
    print(f"\n🎯 Training on {len(tasks)} real ARC tasks")
    print(f"📊 {epochs_per_task} epochs per task, {batch_tasks} tasks per checkpoint")
    
    all_losses = []
    task_losses = {}
    batch_times = []
    
    # Shuffle tasks if not using curriculum
    if not curriculum:
        random.shuffle(tasks)
    
    # Training loop with progress tracking
    num_batches = (len(tasks) - 1) // batch_tasks + 1
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_tasks
        batch_end = min(batch_start + batch_tasks, len(tasks))
        batch_tasks_list = tasks[batch_start:batch_end]
        
        print(f"\n📦 Training batch {batch_idx + 1}/{num_batches}")
        print(f"   Tasks: {batch_start+1}-{batch_end}")
        if curriculum:
            difficulties = [t['difficulty'] for t in batch_tasks_list]
            print(f"   Difficulty range: {min(difficulties):.3f} - {max(difficulties):.3f}")
        
        batch_start_time = time.time()
        batch_losses = []
        
        for task_info in tqdm(batch_tasks_list, desc="Tasks"):
            task_id = task_info['task_id']
            task_data = task_info['data']
            
            try:
                # Adaptive epochs based on difficulty (if curriculum learning)
                task_epochs = epochs_per_task
                if curriculum:
                    difficulty = task_info['difficulty']
                    # Harder tasks get more epochs
                    task_epochs = int(epochs_per_task * (0.7 + 0.6 * difficulty))
                
                # Train on this task
                loss = trainer.train_on_task(task_data, num_epochs=task_epochs)
                batch_losses.append(loss)
                task_losses[task_id] = loss
                
            except Exception as e:
                print(f"❌ Error training on task {task_id}: {e}")
                continue
        
        # Batch summary
        if batch_losses:
            avg_batch_loss = np.mean(batch_losses)
            all_losses.extend(batch_losses)
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            print(f"📊 Batch {batch_idx + 1} avg loss: {avg_batch_loss:.4f}")
            print(f"⏱️  Batch time: {batch_time:.1f}s")
            
            # Adaptive learning rate
            if adaptive_lr and len(all_losses) > 20:
                recent_losses = all_losses[-20:]
                if len(recent_losses) >= 20 and np.std(recent_losses) < 0.1:
                    # If loss plateaued, reduce learning rate
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.95
                    print(f"📉 Adjusted LR to: {trainer.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint regularly
            if (batch_idx + 1) % save_frequency == 0 or batch_idx == num_batches - 1:
                checkpoint_path = exp_dir / f"checkpoint_batch_{batch_idx + 1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'batch': batch_idx + 1,
                    'avg_loss': avg_batch_loss,
                    'task_losses': task_losses,
                    'current_lr': trainer.optimizer.param_groups[0]['lr']
                }, checkpoint_path)
                
                print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Final results and analysis
    if all_losses:
        final_avg_loss = np.mean(all_losses)
        total_time = sum(batch_times)
        
        print(f"\n🎯 Training Complete!")
        print(f"📊 Final average loss: {final_avg_loss:.4f}")
        print(f"📈 Total tasks trained: {len(task_losses)}")
        print(f"⏱️  Total training time: {total_time/3600:.1f} hours")
        print(f"🚄 Average time per task: {total_time/len(task_losses):.1f}s")
        
        # Loss analysis
        losses = list(task_losses.values())
        excellent = len([l for l in losses if l < 0.5])
        good = len([l for l in losses if 0.5 <= l < 1.0])
        fair = len([l for l in losses if 1.0 <= l < 1.5])
        poor = len([l for l in losses if l >= 1.5])
        
        print(f"\n📊 Performance Distribution:")
        print(f"   Excellent (<0.5): {excellent} tasks ({excellent/len(losses)*100:.1f}%)")
        print(f"   Good (0.5-1.0): {good} tasks ({good/len(losses)*100:.1f}%)")
        print(f"   Fair (1.0-1.5): {fair} tasks ({fair/len(losses)*100:.1f}%)")
        print(f"   Poor (>=1.5): {poor} tasks ({poor/len(losses)*100:.1f}%)")
        
        # Save final model
        final_model_path = exp_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'final_avg_loss': final_avg_loss,
            'num_tasks': len(task_losses),
            'task_losses': task_losses,
            'training_time_hours': total_time / 3600,
            'performance_distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            }
        }, final_model_path)
        
        print(f"💾 Final model saved: {final_model_path}")
        
        # Save comprehensive training log
        log_path = exp_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'experiment_name': exp_name,
                'num_tasks': len(task_losses),
                'epochs_per_task': epochs_per_task,
                'curriculum_learning': curriculum,
                'adaptive_lr': adaptive_lr,
                'final_avg_loss': final_avg_loss,
                'training_time_hours': total_time / 3600,
                'task_losses': task_losses,
                'performance_distribution': {
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor
                },
                'model_config': {
                    'grid_size': 30,
                    'num_colors': 10,
                    'embed_dim': 256,
                    'num_fractal_levels': 4,
                    'heads': 8
                }
            }, f, indent=2)
        
        print(f"📋 Training log saved: {log_path}")
        
        return final_model_path
    
    else:
        print("❌ No successful training!")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train FRALA-ARC on full ARC dataset")
    parser.add_argument("--model-path", type=str, help="Path to existing model to continue training")
    parser.add_argument("--arc-data", type=str, default="arc_dataset/data", help="Path to ARC dataset")
    parser.add_argument("--max-tasks", type=int, default=400, help="Maximum number of tasks (400 = all)")
    parser.add_argument("--epochs-per-task", type=int, default=40, help="Base epochs per task")
    parser.add_argument("--batch-tasks", type=int, default=8, help="Tasks per checkpoint batch")
    parser.add_argument("--exp-name", type=str, default="full_arc_training", help="Experiment name")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--no-adaptive-lr", action="store_true", help="Disable adaptive learning rate")
    parser.add_argument("--save-freq", type=int, default=5, help="Save checkpoint every N batches")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    
    args = parser.parse_args()
    
    # Train on full ARC dataset
    final_model_path = train_on_full_arc(
        model_path=args.model_path,
        arc_data_dir=args.arc_data,
        max_tasks=args.max_tasks,
        epochs_per_task=args.epochs_per_task,
        batch_tasks=args.batch_tasks,
        exp_name=args.exp_name,
        curriculum=not args.no_curriculum,
        adaptive_lr=not args.no_adaptive_lr,
        save_frequency=args.save_freq
    )
    
    # Evaluate if requested
    if args.evaluate and final_model_path:
        from train_frala_real_arc import evaluate_after_training
        evaluate_after_training(final_model_path, args.arc_data)

if __name__ == "__main__":
    main() 