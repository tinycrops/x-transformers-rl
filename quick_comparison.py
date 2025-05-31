#!/usr/bin/env python3
"""Quick comparison on different evaluation tasks"""

import torch
import json
import numpy as np
from pathlib import Path
from frala_arc import FRALA_ARC, ARCTrainer

# Load both models
print("📁 Loading models...")
model_synthetic = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
checkpoint_synthetic = torch.load('arc_experiments/arc_demo_quick_final/final_model.pt', weights_only=False)
model_synthetic.load_state_dict(checkpoint_synthetic['model_state_dict'])
trainer_synthetic = ARCTrainer(model_synthetic)

model_real = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
checkpoint_real = torch.load('arc_experiments/real_arc_v1/final_model.pt', weights_only=False)
model_real.load_state_dict(checkpoint_real['model_state_dict'])
trainer_real = ARCTrainer(model_real)

# Test on specific tasks that were in training set vs not
eval_dir = Path('arc_dataset/data/evaluation')
eval_files = list(eval_dir.glob('*.json'))[20:30]  # Different subset

print(f"🧪 Testing on different 10 tasks...")

results = []
for eval_file in eval_files:
    task_id = eval_file.stem
    
    try:
        with open(eval_file) as f:
            task = json.load(f)
        
        # Test both models
        try:
            result_syn = trainer_synthetic.evaluate_task(task)
            acc_syn = result_syn['accuracy']
        except:
            acc_syn = 0.0
        
        try:
            result_real = trainer_real.evaluate_task(task)
            acc_real = result_real['accuracy']
        except:
            acc_real = 0.0
        
        improvement = acc_real - acc_syn
        results.append((task_id, acc_syn, acc_real, improvement))
        
        print(f"  {task_id}: {acc_syn:.3f} → {acc_real:.3f} ({improvement:+.3f})")
        
    except Exception as e:
        print(f"  ❌ {task_id}: Error")

# Summary
accs_syn = [r[1] for r in results]
accs_real = [r[2] for r in results]
improvements = [r[3] for r in results]

print(f"\n📊 Summary:")
print(f"Synthetic model avg: {np.mean(accs_syn):.3f}")
print(f"Real-trained model avg: {np.mean(accs_real):.3f}")
print(f"Average improvement: {np.mean(improvements):.3f}")
print(f"Tasks improved: {sum(1 for imp in improvements if imp > 0.05)}/{len(improvements)}")
print(f"Max improvement: {max(improvements):.3f}") 