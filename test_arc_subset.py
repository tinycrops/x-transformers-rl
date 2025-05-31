#!/usr/bin/env python3
"""Test FRALA-ARC on a subset of real ARC tasks"""

import torch
import json
import numpy as np
from pathlib import Path
from frala_arc import FRALA_ARC, ARCTrainer

def test_arc_subset():
    print("🧪 Testing FRALA-ARC on Real ARC Task Subset")
    print("=" * 60)
    
    # Load the trained model
    print("📁 Loading trained model...")
    model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    checkpoint = torch.load('arc_experiments/arc_demo_quick_final/final_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = ARCTrainer(model)
    
    # Get list of ARC tasks
    arc_dir = Path('arc_dataset/data/evaluation')
    task_files = list(arc_dir.glob('*.json'))[:10]  # Test first 10 tasks
    
    print(f"📊 Testing {len(task_files)} ARC tasks...")
    
    results = []
    
    for i, task_file in enumerate(task_files):
        print(f"\n🧩 Task {i+1}: {task_file.name}")
        
        try:
            # Load task
            with open(task_file) as f:
                task = json.load(f)
            
            # Analyze task structure
            num_train = len(task['train'])
            input_shapes = [f"{len(ex['input'])}x{len(ex['input'][0])}" for ex in task['train']]
            output_shapes = [f"{len(ex['output'])}x{len(ex['output'][0])}" for ex in task['train']]
            test_input_shape = f"{len(task['test'][0]['input'])}x{len(task['test'][0]['input'][0])}"
            test_output_shape = f"{len(task['test'][0]['output'])}x{len(task['test'][0]['output'][0])}"
            
            print(f"  📝 Train examples: {num_train}")
            print(f"  📐 Input shapes: {input_shapes}")
            print(f"  📐 Output shapes: {output_shapes}")
            print(f"  🎯 Test: {test_input_shape} → {test_output_shape}")
            
            # Check for consistency
            unique_input_shapes = set(input_shapes)
            unique_output_shapes = set(output_shapes)
            
            if len(unique_output_shapes) > 1:
                print(f"  ⚠️  Inconsistent output shapes: {unique_output_shapes}")
                # Skip this task for now
                results.append({
                    'task': task_file.name,
                    'status': 'skipped',
                    'reason': 'inconsistent_output_shapes',
                    'accuracy': 0.0
                })
                continue
            
            # Evaluate
            result = trainer.evaluate_task(task)
            accuracy = result['accuracy']
            
            print(f"  ✅ Accuracy: {accuracy:.3f}")
            
            results.append({
                'task': task_file.name,
                'status': 'success',
                'accuracy': accuracy,
                'input_shapes': input_shapes,
                'output_shapes': output_shapes,
                'test_input_shape': test_input_shape,
                'test_output_shape': test_output_shape
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'task': task_file.name,
                'status': 'error',
                'error': str(e),
                'accuracy': 0.0
            })
    
    # Summary
    print(f"\n📊 Results Summary:")
    print("=" * 40)
    
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Successful: {len(successful)}")
    print(f"⚠️  Skipped: {len(skipped)}")
    print(f"❌ Errors: {len(errors)}")
    
    if successful:
        accuracies = [r['accuracy'] for r in successful]
        avg_accuracy = np.mean(accuracies)
        print(f"📈 Average accuracy: {avg_accuracy:.3f}")
        
        # Show individual results
        print(f"\n🎯 Individual Results:")
        for r in successful:
            print(f"  {r['task']}: {r['accuracy']:.3f}")
    
    # Show shape analysis
    if successful:
        print(f"\n📐 Shape Analysis:")
        shape_patterns = {}
        for r in successful:
            pattern = f"{r['test_input_shape']} → {r['test_output_shape']}"
            if pattern not in shape_patterns:
                shape_patterns[pattern] = []
            shape_patterns[pattern].append(r['accuracy'])
        
        for pattern, accs in shape_patterns.items():
            avg_acc = np.mean(accs)
            print(f"  {pattern}: {avg_acc:.3f} ({len(accs)} tasks)")
    
    print(f"\n🎉 Testing completed!")
    return results

if __name__ == "__main__":
    test_arc_subset() 