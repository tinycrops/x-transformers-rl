#!/usr/bin/env python3
"""
Simple test script for FRALA (Fractal Reinforcement Learning Agent)
This verifies that the implementation is working correctly.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_fractal_components():
    """Test individual FRALA components"""
    print("🧪 Testing FRALA Components")
    print("=" * 50)
    
    try:
        from x_transformers_rl.fractal_rl import (
            FractalLevelEmbedding,
            FractalProcessingBlock,
            FractalEncoder,
            FractalWorldModelActorCritic
        )
        print("✅ Imports successful")
        
        # Test 1: FractalLevelEmbedding
        print("\n📍 Testing FractalLevelEmbedding...")
        level_embed = FractalLevelEmbedding(embed_dim=256, max_levels=4)
        embedding = level_embed(2)
        print(f"   Level embedding shape: {embedding.shape}")
        assert embedding.shape == (256,), f"Expected (256,), got {embedding.shape}"
        
        # Test 2: FractalProcessingBlock
        print("\n📍 Testing FractalProcessingBlock...")
        block = FractalProcessingBlock(dim=256, heads=8)
        x = torch.randn(2, 10, 256)  # batch=2, seq=10, dim=256
        global_state = torch.randn(2, 1, 256)
        output = block(x, global_state)
        print(f"   Block output shape: {output.shape}")
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        
        # Test 3: FractalEncoder
        print("\n📍 Testing FractalEncoder...")
        encoder = FractalEncoder(
            input_dim=64,
            embed_dim=256,
            num_levels=3,
            heads=8
        )
        x = torch.randn(2, 16, 64)  # batch=2, seq=16, input_dim=64
        output = encoder(x)
        print(f"   Encoder output shape: {output.shape}")
        assert output.shape == (2, 256), f"Expected (2, 256), got {output.shape}"
        
        # Test with return_all_levels
        output, level_outputs = encoder(x, return_all_levels=True)
        print(f"   Number of level outputs: {len(level_outputs)}")
        assert len(level_outputs) == 3, f"Expected 3 levels, got {len(level_outputs)}"
        
        # Test 4: FractalWorldModelActorCritic
        print("\n📍 Testing FractalWorldModelActorCritic...")
        world_model = FractalWorldModelActorCritic(
            state_dim=64,
            num_actions=8,
            critic_dim_pred=100,
            critic_min_max_value=(-10.0, 10.0),
            embed_dim=256,
            num_fractal_levels=3
        )
        
        state = torch.randn(2, 16, 64)
        raw_actions, values, state_pred, dones, cache = world_model(state)
        
        print(f"   Raw actions shape: {raw_actions.shape}")
        print(f"   Values shape: {values.shape}")
        print(f"   Cache keys: {list(cache.keys())}")
        print(f"   Number of fractal levels in cache: {len(cache['fractal_levels'])}")
        
        # Basic shape assertions
        assert raw_actions.shape[0] == 2, f"Batch size mismatch in actions"
        assert values.shape[0] == 2, f"Batch size mismatch in values"
        assert len(cache['fractal_levels']) == 3, f"Expected 3 fractal levels"
        
        print("✅ All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fractal_agent():
    """Test the complete fractal agent"""
    print("\n🤖 Testing FractalAgent")
    print("=" * 50)
    
    try:
        from x_transformers_rl.fractal_agent import FractalLearner
        
        # Create a simple fractal learner
        learner = FractalLearner(
            state_dim=8,
            num_actions=4,
            reward_range=(-10.0, 10.0),
            num_fractal_levels=2,
            fractal_embed_dim=128,
            fractal_heads=4,
            max_timesteps=100,
            num_episodes_per_update=2,
            lr=0.001
        )
        
        print("✅ FractalLearner created successfully")
        print(f"   Fractal config: {learner.agent.get_fractal_info()}")
        
        # Test analysis functionality
        dummy_state = torch.randn(4, 8)  # batch=4, state_dim=8
        # Ensure dummy state is on the same device as the model
        if hasattr(learner.agent, 'device'):
            dummy_state = dummy_state.to(learner.agent.device)
        analysis = learner.agent.analyze_fractal_representations(dummy_state)
        
        print(f"   Analysis keys: {list(analysis.keys())}")
        if 'representation_diversity' in analysis:
            print(f"   Representation diversity: {analysis['representation_diversity']}")
        
        print("✅ FractalAgent test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_config():
    """Test training configuration creation"""
    print("\n⚙️  Testing Training Configuration")
    print("=" * 50)
    
    try:
        from train_fractal_lander import create_fractal_config
        
        for difficulty in ["easy", "medium", "hard", "extreme"]:
            config = create_fractal_config(difficulty)
            print(f"   {difficulty.upper()} config:")
            print(f"     Levels: {config['num_fractal_levels']}")
            print(f"     Embed dim: {config['fractal_embed_dim']}")
            print(f"     Heads: {config['fractal_heads']}")
            print(f"     Share weights: {config['fractal_share_weights']}")
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_forward_pass():
    """Test a simple forward pass through the complete system"""
    print("\n🔄 Testing Complete Forward Pass")
    print("=" * 50)
    
    try:
        from x_transformers_rl.fractal_rl import FractalWorldModelActorCritic
        
        # Create model
        model = FractalWorldModelActorCritic(
            state_dim=8,
            num_actions=4,
            critic_dim_pred=50,
            critic_min_max_value=(-5.0, 5.0),
            embed_dim=128,
            num_fractal_levels=2,
            heads=4
        )
        
        # Forward pass
        batch_size, seq_len, state_dim = 2, 5, 8
        state = torch.randn(batch_size, seq_len, state_dim)
        
        with torch.no_grad():
            raw_actions, values, _, _, cache = model(state)
        
        print(f"   Input state shape: {state.shape}")
        print(f"   Output actions shape: {raw_actions.shape}")
        print(f"   Output values shape: {values.shape}")
        print(f"   Fractal levels processed: {len(cache.get('fractal_levels', []))}")
        
        # Check for NaN or infinite values
        assert not torch.isnan(raw_actions).any(), "NaN values in actions"
        assert not torch.isnan(values).any(), "NaN values in values"
        assert not torch.isinf(raw_actions).any(), "Infinite values in actions"
        assert not torch.isinf(values).any(), "Infinite values in values"
        
        print("✅ Forward pass test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🌀 FRALA (Fractal Reinforcement Learning Agent) Test Suite")
    print("=" * 70)
    
    tests = [
        test_fractal_components,
        test_fractal_agent,
        test_training_config,
        test_simple_forward_pass
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Fractal Components",
        "Fractal Agent",
        "Training Config",
        "Forward Pass"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! FRALA implementation is working correctly.")
        print("\nTo train a fractal agent, run:")
        print("  python train_fractal_lander.py --difficulty medium --updates 100")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 