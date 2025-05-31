#!/usr/bin/env python3
import torch

# Check if the models are actually different
synthetic = torch.load('arc_experiments/arc_demo_quick_final/final_model.pt', weights_only=False)
real = torch.load('arc_experiments/real_arc_v1/final_model.pt', weights_only=False)

# Compare a few key parameters
syn_params = list(synthetic['model_state_dict'].values())
real_params = list(real['model_state_dict'].values())

print('📊 Model Comparison:')
print(f'Synthetic model params: {len(syn_params)}')
print(f'Real-trained model params: {len(real_params)}')

# Compare first few parameters
diff = torch.norm(syn_params[0] - real_params[0]).item()
print(f'Parameter difference (first layer): {diff:.6f}')

# Check multiple layers
total_diff = 0
for i in range(min(5, len(syn_params))):
    layer_diff = torch.norm(syn_params[i] - real_params[i]).item()
    total_diff += layer_diff
    print(f'Layer {i} difference: {layer_diff:.6f}')

print(f'Total difference: {total_diff:.6f}')

if total_diff < 1e-5:
    print('⚠️  Models appear very similar!')
else:
    print('✅ Models are different')
    
# Check training info
print(f'\nSynthetic model info:')
if 'final_avg_loss' in synthetic:
    print(f'  Final loss: {synthetic["final_avg_loss"]:.4f}')
if 'num_tasks' in synthetic:
    print(f'  Num tasks: {synthetic["num_tasks"]}')

print(f'\nReal-trained model info:')
if 'final_avg_loss' in real:
    print(f'  Final loss: {real["final_avg_loss"]:.4f}')
if 'num_tasks' in real:
    print(f'  Num tasks: {real["num_tasks"]}') 