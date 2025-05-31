#!/usr/bin/env python3
"""
Train FRALA-ARC on Real ARC Training Dataset

This script trains the FRALA-ARC model on the official ARC training dataset
to improve performance on real ARC tasks.
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

def load_arc_training_tasks(arc_data_dir: str, max_tasks: int = None):
    """Load real ARC training tasks."""
    training_dir = Path(arc_data_dir) / "training"
    task_files = list(training_dir.glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    tasks = []
    print(f"📚 Loading {len(task_files)} ARC training tasks...")
    
    for task_file in tqdm(task_files):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            
            # Validate task structure
            if 'train' in task_data and len(task_data['train']) > 0:
                # Check grid sizes (skip tasks with very large grids)
                max_size = 25  # Reasonable limit for training
                valid_task = True
                
                for example in task_data['train']:
                    input_h, input_w = len(example['input']), len(example['input'][0])
                    output_h, output_w = len(example['output']), len(example['output'][0])
                    
                    if input_h > max_size or input_w > max_size or output_h > max_size or output_w > max_size:
                        valid_task = False
                        break
                
                if valid_task:
                    tasks.append({
                        'task_id': task_file.stem,
                        'data': task_data
                    })
        
        except Exception as e:
            print(f"❌ Error loading {task_file}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(tasks)} valid ARC tasks")
    return tasks

def train_on_real_arc(
    model_path: str = None,
    arc_data_dir: str = "arc_dataset/data",
    max_tasks: int = 200,  # Train on subset first
    epochs_per_task: int = 50,  # Fewer epochs per task
    batch_tasks: int = 10,  # Process multiple tasks per batch
    exp_name: str = "real_arc_training"
):
    """Train FRALA-ARC on real ARC training data."""
    
    print("🚀 FRALA-ARC Real ARC Training")
    print("=" * 50)
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"📁 Loading existing model from {model_path}")
        model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("🆕 Creating new model")
        model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    
    trainer = ARCTrainer(model, learning_rate=0.0005)  # Lower learning rate for real data
    
    # Load real ARC tasks
    tasks = load_arc_training_tasks(arc_data_dir, max_tasks)
    
    if not tasks:
        print("❌ No valid tasks found!")
        return
    
    # Create experiment directory
    exp_dir = Path("arc_experiments") / exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    print(f"\n🎯 Training on {len(tasks)} real ARC tasks")
    print(f"📊 {epochs_per_task} epochs per task, {batch_tasks} tasks per checkpoint")
    
    all_losses = []
    task_losses = {}
    
    # Shuffle tasks for better learning
    random.shuffle(tasks)
    
    for batch_start in range(0, len(tasks), batch_tasks):
        batch_end = min(batch_start + batch_tasks, len(tasks))
        batch_tasks_list = tasks[batch_start:batch_end]
        
        print(f"\n📦 Training batch {batch_start//batch_tasks + 1}/{(len(tasks)-1)//batch_tasks + 1}")
        print(f"   Tasks: {batch_start+1}-{batch_end}")
        
        batch_losses = []
        
        for task_info in tqdm(batch_tasks_list, desc="Tasks"):
            task_id = task_info['task_id']
            task_data = task_info['data']
            
            try:
                # Train on this task
                loss = trainer.train_on_task(task_data, num_epochs=epochs_per_task)
                batch_losses.append(loss)
                task_losses[task_id] = loss
                
            except Exception as e:
                print(f"❌ Error training on task {task_id}: {e}")
                continue
        
        if batch_losses:
            avg_batch_loss = np.mean(batch_losses)
            all_losses.extend(batch_losses)
            
            print(f"📊 Batch {batch_start//batch_tasks + 1} avg loss: {avg_batch_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = exp_dir / f"checkpoint_batch_{batch_start//batch_tasks + 1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'batch': batch_start//batch_tasks + 1,
                'avg_loss': avg_batch_loss,
                'task_losses': task_losses
            }, checkpoint_path)
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Final results
    if all_losses:
        final_avg_loss = np.mean(all_losses)
        print(f"\n🎯 Training Complete!")
        print(f"📊 Final average loss: {final_avg_loss:.4f}")
        print(f"📈 Total tasks trained: {len(task_losses)}")
        
        # Save final model
        final_model_path = exp_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'final_avg_loss': final_avg_loss,
            'num_tasks': len(task_losses),
            'task_losses': task_losses
        }, final_model_path)
        
        print(f"💾 Final model saved: {final_model_path}")
        
        # Save training log
        log_path = exp_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'experiment_name': exp_name,
                'num_tasks': len(task_losses),
                'epochs_per_task': epochs_per_task,
                'final_avg_loss': final_avg_loss,
                'task_losses': task_losses,
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

def evaluate_after_training(model_path: str, arc_data_dir: str = "arc_dataset/data"):
    """Evaluate the trained model on evaluation tasks."""
    
    print(f"\n🔍 Evaluating trained model: {model_path}")
    
    # Load trained model
    model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = ARCTrainer(model)
    
    # Load evaluation tasks (subset)
    eval_dir = Path(arc_data_dir) / "evaluation"
    eval_files = list(eval_dir.glob("*.json"))[:20]  # Test on 20 tasks
    
    print(f"📊 Evaluating on {len(eval_files)} tasks...")
    
    results = []
    
    for eval_file in tqdm(eval_files):
        try:
            with open(eval_file) as f:
                task = json.load(f)
            
            result = trainer.evaluate_task(task)
            results.append({
                'task': eval_file.stem,
                'accuracy': result['accuracy']
            })
            
        except Exception as e:
            print(f"❌ Error evaluating {eval_file.stem}: {e}")
            results.append({
                'task': eval_file.stem,
                'accuracy': 0.0
            })
    
    # Summary
    accuracies = [r['accuracy'] for r in results]
    avg_accuracy = np.mean(accuracies)
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Average accuracy: {avg_accuracy:.3f}")
    print(f"   Best accuracy: {max(accuracies):.3f}")
    print(f"   Tasks with >0% accuracy: {sum(1 for acc in accuracies if acc > 0)}/{len(accuracies)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train FRALA-ARC on real ARC data")
    parser.add_argument("--model-path", type=str, help="Path to existing model to continue training")
    parser.add_argument("--arc-data", type=str, default="arc_dataset/data", help="Path to ARC dataset")
    parser.add_argument("--max-tasks", type=int, default=200, help="Maximum number of tasks to train on")
    parser.add_argument("--epochs-per-task", type=int, default=50, help="Epochs per task")
    parser.add_argument("--batch-tasks", type=int, default=10, help="Tasks per checkpoint batch")
    parser.add_argument("--exp-name", type=str, default="real_arc_training", help="Experiment name")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    
    args = parser.parse_args()
    
    # Train on real ARC data
    final_model_path = train_on_real_arc(
        model_path=args.model_path,
        arc_data_dir=args.arc_data,
        max_tasks=args.max_tasks,
        epochs_per_task=args.epochs_per_task,
        batch_tasks=args.batch_tasks,
        exp_name=args.exp_name
    )
    
    # Evaluate if requested
    if args.evaluate and final_model_path:
        evaluate_after_training(final_model_path, args.arc_data)

if __name__ == "__main__":
    main() 