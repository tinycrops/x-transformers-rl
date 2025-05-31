#!/usr/bin/env python3
"""Test FRALA-ARC on a single real ARC task"""

import torch
import json
from frala_arc import FRALA_ARC, ARCTrainer

def test_single_arc_task():
    print("🧪 Testing FRALA-ARC on Real ARC Task")
    print("=" * 50)
    
    # Load the trained model
    print("📁 Loading trained model...")
    model = FRALA_ARC(grid_size=30, num_colors=10, embed_dim=256, num_fractal_levels=4, heads=8)
    checkpoint = torch.load('arc_experiments/arc_demo_quick_final/final_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = ARCTrainer(model)
    
    # Test on one ARC task
    print("📋 Loading ARC task...")
    with open('arc_dataset/data/evaluation/00576224.json') as f:
        task = json.load(f)
    
    print("📊 Task structure:")
    print(f"  Train examples: {len(task['train'])}")
    print(f"  Test examples: {len(task['test'])}")
    print(f"  First input shape: {len(task['train'][0]['input'])}x{len(task['train'][0]['input'][0])}")
    print(f"  First output shape: {len(task['train'][0]['output'])}x{len(task['train'][0]['output'][0])}")
    
    # Show the actual task
    print("\n🔍 Task pattern (first example):")
    print("Input:")
    for row in task['train'][0]['input']:
        print(f"  {row}")
    print("Output:")
    for row in task['train'][0]['output']:
        print(f"  {row}")
    
    # Evaluate
    print("\n🔄 Evaluating...")
    result = trainer.evaluate_task(task)
    print(f"✅ Accuracy: {result['accuracy']:.3f}")
    
    # Show prediction
    prediction = result['predicted_grid']
    target = result['target_grid']
    
    print(f"\n📏 Prediction shape: {prediction.shape}")
    print(f"📏 Target shape: {target.shape}")
    
    print("\n🎯 Test prediction:")
    for row in prediction:
        print(f"  {row.tolist()}")
    
    print("\n🎯 Target output:")
    for row in target:
        print(f"  {row.tolist()}")
    
    print("\n🎉 Test completed successfully!")
    return result

if __name__ == "__main__":
    test_single_arc_task() 