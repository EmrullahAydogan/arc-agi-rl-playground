#!/usr/bin/env python3
"""
Train Imitation Learning Agent from Human Demonstrations
Behavioral Cloning: Learn to mimic human puzzle-solving strategies
"""
import sys
import argparse
from pathlib import Path
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.demonstration_buffer import DemonstrationBuffer
from src.agents.imitation_agent import ImitationAgent
from src.utils.checkpoint_manager import CheckpointManager


def train_imitation_agent(
    demo_file: str = "demonstrations.pkl",
    checkpoint_dir: str = "checkpoints/imitation",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_interval: int = 10,
    only_successful: bool = True,
    accuracy_threshold: float = 90.0,
    device: str = "auto"
):
    """
    Train imitation learning agent from demonstrations

    Args:
        demo_file: Demonstration file to load
        checkpoint_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_interval: Save checkpoint every N epochs
        only_successful: Only train on successful demonstrations
        accuracy_threshold: Minimum accuracy for successful demos
        device: "cuda", "cpu", or "auto"
    """
    print("=" * 80)
    print(">>> IMITATION LEARNING TRAINING <<<")
    print("=" * 80)
    print(f"[CONFIG] Demo file: {demo_file}")
    print(f"[CONFIG] Checkpoint dir: {checkpoint_dir}")
    print(f"[CONFIG] Epochs: {num_epochs}")
    print(f"[CONFIG] Batch size: {batch_size}")
    print(f"[CONFIG] Learning rate: {learning_rate}")
    print(f"[CONFIG] Only successful: {only_successful}")
    print(f"[CONFIG] Accuracy threshold: {accuracy_threshold}%")
    print("=" * 80)
    print()

    # Load demonstration buffer
    print("[LOADING] Loading demonstrations...")
    demo_buffer = DemonstrationBuffer(save_dir="demonstrations")
    demo_buffer.load(demo_file)

    if demo_buffer.total_demonstrations == 0:
        print("[ERROR] No demonstrations found!")
        print(f"[HELP] Please run 'python demo_playground.py' first to record demonstrations")
        return

    # Print summary
    demo_buffer.print_summary()

    # Check if we have enough data
    pairs = demo_buffer.get_all_state_action_pairs(
        only_successful=only_successful,
        accuracy_threshold=accuracy_threshold
    )

    if len(pairs) == 0:
        print(f"[ERROR] No training data available!")
        if only_successful:
            print(f"[HELP] Try setting --all-demos to include all demonstrations, not just successful ones")
        return

    print(f"\n[DATA] Total state-action pairs available: {len(pairs)}")
    print(f"[DATA] This gives us {len(pairs) // batch_size} batches per epoch")
    print()

    # Create imitation agent
    print("[SETUP] Creating imitation learning agent...")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = ImitationAgent(
        action_space_size=9004,  # 30*30*10 + 4
        max_grid_size=30,
        num_colors=10,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Training loop
    print("\n[TRAINING] Starting training...")
    print("=" * 80)

    best_accuracy = 0.0
    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train for one epoch
        # We'll do multiple batches per epoch
        batches_per_epoch = max(1, len(pairs) // batch_size)

        epoch_metrics = []
        for batch_idx in range(batches_per_epoch):
            metrics = agent.update(
                demonstration_buffer=demo_buffer,
                batch_size=batch_size,
                num_epochs=1,
                only_successful=only_successful
            )
            epoch_metrics.append(metrics)

        # Average metrics
        avg_loss = sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_accuracy = sum(m['accuracy'] for m in epoch_metrics) / len(epoch_metrics)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {avg_accuracy:5.1f}% | "
              f"Time: {epoch_time:.2f}s")

        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = Path(checkpoint_dir) / f"imitation_epoch_{epoch}.pt"
            agent.save(str(checkpoint_path))
            print(f"   [CHECKPOINT] Saved to {checkpoint_path}")

        # Save best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_path = Path(checkpoint_dir) / "imitation_best.pt"
            agent.save(str(best_path))
            print(f"   [BEST] New best accuracy: {best_accuracy:.1f}% -> Saved to {best_path}")

    # Training complete
    training_time = time.time() - training_start

    print("=" * 80)
    print("[TRAINING] Training complete!")
    print(f"   Total time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"   Best accuracy: {best_accuracy:.1f}%")
    print(f"   Final loss: {avg_loss:.4f}")
    print(f"   Training steps: {agent.training_steps}")

    # Save final model
    final_path = Path(checkpoint_dir) / "imitation_final.pt"
    agent.save(str(final_path))
    print(f"   Final model saved to {final_path}")

    print("\n[NEXT STEPS] Use trained agent with:")
    print(f"   python playground.py --agent imitation --checkpoint {best_path}")
    print("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train Imitation Learning Agent from Human Demonstrations"
    )

    parser.add_argument(
        '--demo-file',
        type=str,
        default='demonstrations.pkl',
        help='Demonstration file to load'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/imitation',
        help='Directory to save checkpoints'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    parser.add_argument(
        '--all-demos',
        action='store_true',
        help='Train on all demonstrations, not just successful ones'
    )

    parser.add_argument(
        '--accuracy-threshold',
        type=float,
        default=90.0,
        help='Minimum accuracy for successful demonstrations'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training'
    )

    args = parser.parse_args()

    try:
        train_imitation_agent(
            demo_file=args.demo_file,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_interval=args.save_interval,
            only_successful=not args.all_demos,
            accuracy_threshold=args.accuracy_threshold,
            device=args.device
        )
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
