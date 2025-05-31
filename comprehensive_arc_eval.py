#!/usr/bin/env python3
"""
Comprehensive evaluation of FRALA-ARC on real ARC tasks
"""

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from frala_arc import FRALA_ARC, ARCTrainer

def comprehensive_evaluation():
    print("🔬 Comprehensive FRALA-ARC Evaluation")
    print("=" * 50)
    
    # Load both models for comparison
    print("📁 Loading models...")
    
    # Original synthetic-trained model
    model_synthetic = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    checkpoint_synthetic = torch.load('arc_experiments/arc_demo_quick_final/final_model.pt', weights_only=False)
    model_synthetic.load_state_dict(checkpoint_synthetic['model_state_dict'])
    trainer_synthetic = ARCTrainer(model_synthetic)
    
    # Real-ARC-trained model
    model_real = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    checkpoint_real = torch.load('arc_experiments/real_arc_v1/final_model.pt', weights_only=False)
    model_real.load_state_dict(checkpoint_real['model_state_dict'])
    trainer_real = ARCTrainer(model_real)
    
    # Load evaluation tasks
    eval_dir = Path('arc_dataset/data/evaluation')
    eval_files = list(eval_dir.glob('*.json'))[:50]  # Test on more tasks
    
    print(f"📊 Evaluating on {len(eval_files)} tasks...")
    
    results_synthetic = []
    results_real = []
    
    for eval_file in tqdm(eval_files):
        task_id = eval_file.stem
        
        try:
            with open(eval_file) as f:
                task = json.load(f)
            
            # Evaluate synthetic model
            try:
                result_syn = trainer_synthetic.evaluate_task(task)
                acc_syn = result_syn['accuracy']
            except:
                acc_syn = 0.0
            
            # Evaluate real-ARC model
            try:
                result_real_task = trainer_real.evaluate_task(task)
                acc_real = result_real_task['accuracy']
            except:
                acc_real = 0.0
            
            results_synthetic.append({
                'task': task_id,
                'accuracy': acc_syn
            })
            
            results_real.append({
                'task': task_id,
                'accuracy': acc_real
            })
            
        except Exception as e:
            print(f"❌ Error with task {task_id}: {e}")
            continue
    
    # Analysis
    print(f"\n📈 Comprehensive Results:")
    print("=" * 50)
    
    accs_syn = [r['accuracy'] for r in results_synthetic]
    accs_real = [r['accuracy'] for r in results_real]
    
    print(f"🤖 Synthetic-Only Model:")
    print(f"   Average accuracy: {np.mean(accs_syn):.3f}")
    print(f"   Median accuracy: {np.median(accs_syn):.3f}")
    print(f"   Best accuracy: {np.max(accs_syn):.3f}")
    print(f"   Tasks with >0% accuracy: {sum(1 for acc in accs_syn if acc > 0)}/{len(accs_syn)}")
    print(f"   Tasks with >50% accuracy: {sum(1 for acc in accs_syn if acc > 0.5)}/{len(accs_syn)}")
    
    print(f"\n🚀 Real-ARC-Trained Model:")
    print(f"   Average accuracy: {np.mean(accs_real):.3f}")
    print(f"   Median accuracy: {np.median(accs_real):.3f}")
    print(f"   Best accuracy: {np.max(accs_real):.3f}")
    print(f"   Tasks with >0% accuracy: {sum(1 for acc in accs_real if acc > 0)}/{len(accs_real)}")
    print(f"   Tasks with >50% accuracy: {sum(1 for acc in accs_real if acc > 0.5)}/{len(accs_real)}")
    
    # Improvement analysis
    improvements = [real - syn for syn, real in zip(accs_syn, accs_real)]
    improved_tasks = sum(1 for imp in improvements if imp > 0.1)
    
    print(f"\n💡 Improvement Analysis:")
    print(f"   Average improvement: {np.mean(improvements):.3f}")
    print(f"   Tasks with >10% improvement: {improved_tasks}/{len(improvements)}")
    print(f"   Largest improvement: {np.max(improvements):.3f}")
    
    # Task-by-task comparison for biggest improvements
    print(f"\n🏆 Top 10 Most Improved Tasks:")
    combined_results = [(results_synthetic[i]['task'], accs_syn[i], accs_real[i], improvements[i]) 
                       for i in range(len(improvements))]
    combined_results.sort(key=lambda x: x[3], reverse=True)
    
    for i, (task_id, acc_syn, acc_real, improvement) in enumerate(combined_results[:10]):
        print(f"   {i+1}. {task_id}: {acc_syn:.3f} → {acc_real:.3f} (+{improvement:.3f})")
    
    # Save detailed results
    detailed_results = {
        'num_tasks': len(eval_files),
        'synthetic_model': {
            'avg_accuracy': float(np.mean(accs_syn)),
            'median_accuracy': float(np.median(accs_syn)),
            'best_accuracy': float(np.max(accs_syn)),
            'tasks_with_progress': sum(1 for acc in accs_syn if acc > 0),
            'tasks_with_success': sum(1 for acc in accs_syn if acc > 0.5)
        },
        'real_trained_model': {
            'avg_accuracy': float(np.mean(accs_real)),
            'median_accuracy': float(np.median(accs_real)),
            'best_accuracy': float(np.max(accs_real)),
            'tasks_with_progress': sum(1 for acc in accs_real if acc > 0),
            'tasks_with_success': sum(1 for acc in accs_real if acc > 0.5)
        },
        'improvement': {
            'avg_improvement': float(np.mean(improvements)),
            'tasks_improved': improved_tasks,
            'max_improvement': float(np.max(improvements))
        },
        'task_results': [
            {
                'task_id': task_id,
                'synthetic_accuracy': float(acc_syn),
                'real_trained_accuracy': float(acc_real),
                'improvement': float(improvement)
            }
            for task_id, acc_syn, acc_real, improvement in combined_results
        ]
    }
    
    eval_results_path = Path('arc_experiments/real_arc_v1/comprehensive_evaluation.json')
    with open(eval_results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved: {eval_results_path}")
    print(f"\n🎉 Evaluation complete!")

if __name__ == "__main__":
    comprehensive_evaluation() 