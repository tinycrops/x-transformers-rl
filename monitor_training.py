#!/usr/bin/env python3
"""Monitor FRALA-ARC training progress"""

import json
import time
from pathlib import Path
import argparse

def monitor_training(exp_name="full_arc_curriculum_v1", refresh_interval=30):
    """Monitor training progress in real-time."""
    exp_dir = Path("arc_experiments") / exp_name
    
    print(f"🔍 Monitoring training: {exp_name}")
    print(f"📂 Experiment directory: {exp_dir}")
    print("=" * 60)
    
    last_batch = 0
    start_time = time.time()
    
    try:
        while True:
            # Check for new checkpoints
            checkpoints = list(exp_dir.glob("checkpoint_batch_*.pt"))
            if checkpoints:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                batch_num = int(latest_checkpoint.stem.split('_')[-1])
                
                if batch_num > last_batch:
                    print(f"\n🔄 New checkpoint: Batch {batch_num}")
                    
                    # Load and display progress
                    try:
                        import torch
                        checkpoint = torch.load(latest_checkpoint, weights_only=False)
                        
                        current_loss = checkpoint.get('avg_loss', 'N/A')
                        current_lr = checkpoint.get('current_lr', 'N/A')
                        num_tasks_completed = len(checkpoint.get('task_losses', {}))
                        
                        elapsed_time = time.time() - start_time
                        
                        print(f"   📊 Average loss: {current_loss:.4f}")
                        print(f"   📈 Learning rate: {current_lr:.6f}")
                        print(f"   ✅ Tasks completed: {num_tasks_completed}")
                        print(f"   ⏱️  Elapsed time: {elapsed_time/3600:.1f} hours")
                        
                        # Estimate completion
                        if num_tasks_completed > 0:
                            tasks_per_hour = num_tasks_completed / (elapsed_time / 3600)
                            remaining_tasks = 400 - num_tasks_completed
                            eta_hours = remaining_tasks / tasks_per_hour if tasks_per_hour > 0 else 0
                            
                            print(f"   🚄 Speed: {tasks_per_hour:.1f} tasks/hour")
                            print(f"   🎯 ETA: {eta_hours:.1f} hours")
                        
                        last_batch = batch_num
                        
                    except Exception as e:
                        print(f"   ❌ Error reading checkpoint: {e}")
            
            # Check if training is complete
            final_model = exp_dir / "final_model.pt"
            if final_model.exists():
                print(f"\n🎉 Training completed!")
                
                # Show final results
                try:
                    log_file = exp_dir / "training_log.json"
                    if log_file.exists():
                        with open(log_file) as f:
                            log = json.load(f)
                        
                        print(f"📊 Final Results:")
                        print(f"   Total tasks: {log.get('num_tasks', 'N/A')}")
                        print(f"   Final loss: {log.get('final_avg_loss', 'N/A'):.4f}")
                        print(f"   Training time: {log.get('training_time_hours', 'N/A'):.1f} hours")
                        
                        if 'performance_distribution' in log:
                            perf = log['performance_distribution']
                            total = sum(perf.values())
                            print(f"   Performance breakdown:")
                            print(f"     Excellent: {perf.get('excellent', 0)}/{total} ({perf.get('excellent', 0)/total*100:.1f}%)")
                            print(f"     Good: {perf.get('good', 0)}/{total} ({perf.get('good', 0)/total*100:.1f}%)")
                            print(f"     Fair: {perf.get('fair', 0)}/{total} ({perf.get('fair', 0)/total*100:.1f}%)")
                            print(f"     Poor: {perf.get('poor', 0)}/{total} ({perf.get('poor', 0)/total*100:.1f}%)")
                
                except Exception as e:
                    print(f"   ❌ Error reading final results: {e}")
                
                break
            
            # Wait before next check
            print(f"⏳ Waiting {refresh_interval}s... (Ctrl+C to stop monitoring)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")

def show_current_status(exp_name="full_arc_curriculum_v1"):
    """Show current training status without continuous monitoring."""
    exp_dir = Path("arc_experiments") / exp_name
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    print(f"📊 Current status for: {exp_name}")
    print("=" * 50)
    
    # Check for checkpoints
    checkpoints = list(exp_dir.glob("checkpoint_batch_*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        batch_num = int(latest_checkpoint.stem.split('_')[-1])
        
        try:
            import torch
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            
            print(f"📦 Latest batch: {batch_num}")
            print(f"📊 Average loss: {checkpoint.get('avg_loss', 'N/A'):.4f}")
            print(f"📈 Learning rate: {checkpoint.get('current_lr', 'N/A'):.6f}")
            print(f"✅ Tasks completed: {len(checkpoint.get('task_losses', {}))}")
            
        except Exception as e:
            print(f"❌ Error reading checkpoint: {e}")
    
    # Check if completed
    final_model = exp_dir / "final_model.pt"
    if final_model.exists():
        print("🎉 Training completed!")
        
        log_file = exp_dir / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    log = json.load(f)
                print(f"📊 Final loss: {log.get('final_avg_loss', 'N/A'):.4f}")
                print(f"⏱️  Total time: {log.get('training_time_hours', 'N/A'):.1f} hours")
            except:
                pass
    else:
        print("🔄 Training in progress...")

def main():
    parser = argparse.ArgumentParser(description="Monitor FRALA-ARC training")
    parser.add_argument("--exp-name", type=str, default="full_arc_curriculum_v1", help="Experiment name")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_training(args.exp_name, args.interval)
    else:
        show_current_status(args.exp_name)

if __name__ == "__main__":
    main() 