#!/usr/bin/env python3
import json

with open('arc_experiments/real_arc_v1/training_log.json') as f:
    log = json.load(f)

# Sort tasks by loss (ascending - best performance)
sorted_tasks = sorted(log['task_losses'].items(), key=lambda x: x[1])

print('🏆 Top 10 Best Performing Tasks (lowest loss):')
for i, (task_id, loss) in enumerate(sorted_tasks[:10]):
    print(f'   {i+1}. {task_id}: {loss:.3f}')

print(f'\n📊 Performance Distribution:')
losses = list(log['task_losses'].values())
print(f'   Best loss: {min(losses):.3f}')
print(f'   Worst loss: {max(losses):.3f}') 
print(f'   Average loss: {sum(losses)/len(losses):.3f}')
print(f'   Median loss: {sorted(losses)[len(losses)//2]:.3f}')

print(f'\n🎯 Loss Categories:')
excellent = [l for l in losses if l < 0.5]
good = [l for l in losses if 0.5 <= l < 1.0]
fair = [l for l in losses if 1.0 <= l < 1.5]
poor = [l for l in losses if l >= 1.5]

print(f'   Excellent (<0.5): {len(excellent)} tasks')
print(f'   Good (0.5-1.0): {len(good)} tasks')
print(f'   Fair (1.0-1.5): {len(fair)} tasks')
print(f'   Poor (>=1.5): {len(poor)} tasks') 