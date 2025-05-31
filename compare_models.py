#!/usr/bin/env python3
"""Compare different FRALA-ARC model versions"""

import torch
import json
import numpy as np
from pathlib import Path
from frala_arc import FRALA_ARC, ARCTrainer
from tqdm import tqdm

def compare_models():
    """Compare all available FRALA-ARC models."""
    
    print("🔬 FRALA-ARC Model Comparison")
    print("=" * 60)
    
    # Define model paths and info
    models = {
        "Synthetic Only": "arc_experiments/arc_demo_quick_final/final_model.pt",
        "Real ARC v1 (96 tasks)": "arc_experiments/real_arc_v1/final_model.pt", 
        "Real ARC v2 (143 tasks)": "arc_experiments/real_arc_v2_extended/final_model.pt",
        "Full Dataset v1 (400 tasks)": "arc_experiments/full_arc_curriculum_v1/final_model.pt"
    }
    
    # Check which models exist
    available_models = {}
    for name, path in models.items():
        if Path(path).exists():
            available_models[name] = path
            print(f"✅ {name}: Found")
        else:
            print(f"⏳ {name}: Not available yet")
    
    if len(available_models) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    print(f"\n📊 Comparing {len(available_models)} models...")
    
    # Load evaluation tasks
    eval_dir = Path('arc_dataset/data/evaluation')
    eval_files = list(eval_dir.glob('*.json'))[:30]  # Test on 30 tasks
    
    print(f"🧪 Evaluating on {len(eval_files)} tasks...")
    
    results = {}
    
    # Evaluate each model
    for model_name, model_path in available_models.items():
        print(f"\n🔄 Evaluating: {model_name}")
        
        try:
            # Load model
            model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer = ARCTrainer(model)
            
            # Get training info
            training_info = {
                'final_loss': checkpoint.get('final_avg_loss', 'N/A'),
                'num_tasks': checkpoint.get('num_tasks', 'N/A'),
                'training_time': checkpoint.get('training_time_hours', 'N/A')
            }
            
            # Evaluate on test tasks
            accuracies = []
            for eval_file in tqdm(eval_files, desc="Tasks"):
                try:
                    with open(eval_file) as f:
                        task = json.load(f)
                    
                    result = trainer.evaluate_task(task)
                    accuracies.append(result['accuracy'])
                    
                except:
                    accuracies.append(0.0)
            
            results[model_name] = {
                'accuracies': accuracies,
                'avg_accuracy': np.mean(accuracies),
                'median_accuracy': np.median(accuracies),
                'best_accuracy': np.max(accuracies),
                'tasks_with_progress': sum(1 for acc in accuracies if acc > 0),
                'tasks_with_success': sum(1 for acc in accuracies if acc > 0.5),
                'training_info': training_info
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            results[model_name] = None
    
    # Display comparison
    print(f"\n📈 Model Comparison Results")
    print("=" * 80)
    
    # Create comparison table
    print(f"{'Model':<25} {'Avg Acc':<10} {'Med Acc':<10} {'Best':<8} {'Progress':<10} {'Success':<8} {'Train Loss':<12}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result is None:
            print(f"{model_name:<25} {'ERROR':<10}")
            continue
            
        avg_acc = result['avg_accuracy']
        med_acc = result['median_accuracy']
        best_acc = result['best_accuracy']
        progress = f"{result['tasks_with_progress']}/{len(eval_files)}"
        success = f"{result['tasks_with_success']}/{len(eval_files)}"
        train_loss = result['training_info']['final_loss']
        train_loss_str = f"{train_loss:.3f}" if isinstance(train_loss, (int, float)) else str(train_loss)
        
        print(f"{model_name:<25} {avg_acc:<10.3f} {med_acc:<10.3f} {best_acc:<8.3f} {progress:<10} {success:<8} {train_loss_str:<12}")
    
    # Improvement analysis
    print(f"\n💡 Improvement Analysis")
    print("-" * 40)
    
    # Find baseline (synthetic model)
    baseline_name = "Synthetic Only"
    if baseline_name in results and results[baseline_name] is not None:
        baseline_acc = results[baseline_name]['avg_accuracy']
        
        for model_name, result in results.items():
            if result is None or model_name == baseline_name:
                continue
                
            improvement = result['avg_accuracy'] - baseline_acc
            improvement_pct = (improvement / baseline_acc) * 100
            
            print(f"{model_name}:")
            print(f"  Improvement: +{improvement:.3f} (+{improvement_pct:.1f}%)")
            print(f"  Training tasks: {result['training_info']['num_tasks']}")
    
    # Best performing tasks
    print(f"\n🏆 Best Performing Tasks by Model")
    print("-" * 40)
    
    task_names = [f.stem for f in eval_files]
    
    for i, task_name in enumerate(task_names[:10]):  # Show top 10 tasks
        best_model = None
        best_acc = 0
        
        for model_name, result in results.items():
            if result is None:
                continue
            if result['accuracies'][i] > best_acc:
                best_acc = result['accuracies'][i]
                best_model = model_name
        
        if best_acc > 0.5:  # Only show well-performing tasks
            print(f"  {task_name}: {best_model} ({best_acc:.3f})")
    
    # Save detailed comparison
    comparison_path = Path("arc_experiments/model_comparison.json")
    comparison_data = {
        'evaluation_tasks': len(eval_files),
        'models': results,
        'task_names': task_names
    }
    
    with open(comparison_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        import json
        json.dump(comparison_data, f, indent=2, default=convert_numpy)
    
    print(f"\n💾 Detailed comparison saved: {comparison_path}")
    print(f"🎉 Comparison complete!")

if __name__ == "__main__":
    compare_models() 