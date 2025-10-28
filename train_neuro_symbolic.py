#!/usr/bin/env python3
"""
Train Neuro-Symbolic Agent
Complete reasoning system combining neural and symbolic AI
"""
import sys
import argparse
from pathlib import Path
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.demonstration_buffer import DemonstrationBuffer
from src.agents.neuro_symbolic_agent import NeuroSymbolicAgent


def train_neuro_symbolic(
    demo_file: str = "demonstrations.pkl",
    checkpoint_dir: str = "checkpoints/neuro_symbolic",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_transformer: bool = True,
    use_hierarchical: bool = True,
    mode: str = 'hybrid',
    device: str = "auto"
):
    """
    Train neuro-symbolic agent

    Args:
        demo_file: Demonstration file
        checkpoint_dir: Checkpoint directory
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_transformer: Enable transformer attention
        use_hierarchical: Enable hierarchical policy
        mode: 'transformer', 'hierarchical', or 'hybrid'
        device: 'cuda', 'cpu', or 'auto'
    """
    print("=" * 80)
    print(">>> NEURO-SYMBOLIC AGENT TRAINING <<<")
    print("=" * 80)
    print(f"[CONFIG] Demo file: {demo_file}")
    print(f"[CONFIG] Mode: {mode}")
    print(f"[CONFIG] Transformer: {use_transformer}")
    print(f"[CONFIG] Hierarchical: {use_hierarchical}")
    print(f"[CONFIG] Epochs: {num_epochs}")
    print(f"[CONFIG] Batch size: {batch_size}")
    print("=" * 80)
    print()

    # Load demonstrations
    print("[LOADING] Loading demonstrations...")
    demo_buffer = DemonstrationBuffer(save_dir="demonstrations")
    demo_buffer.load(demo_file)

    if demo_buffer.total_demonstrations == 0:
        print("[ERROR] No demonstrations found!")
        print("[HELP] Run 'python demo_playground.py' first")
        return

    demo_buffer.print_summary()

    # Device selection
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create agent
    print("\n[SETUP] Creating neuro-symbolic agent...")
    agent = NeuroSymbolicAgent(
        action_space_size=9004,
        grid_size=30,
        num_colors=10,
        use_transformer=use_transformer,
        use_hierarchical=use_hierarchical,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    # Set mode
    agent.mode = mode

    # Training
    print("\n[TRAINING] Starting training...")
    print("=" * 80)

    start_time = time.time()

    metrics = agent.train_from_demonstrations(
        demo_buffer=demo_buffer,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    training_time = time.time() - start_time

    print("=" * 80)
    print("[TRAINING] Training complete!")
    print(f"  Total time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"  Average loss: {metrics['avg_loss']:.4f}")
    print(f"  Training steps: {metrics['training_steps']}")

    # Save final model
    final_path = Path(checkpoint_dir) / "neuro_symbolic_final.pt"
    agent.save(str(final_path))
    print(f"  Saved to: {final_path}")

    print("\n[NEXT STEPS]")
    print("  Test the agent:")
    print(f"    python playground.py --agent neuro_symbolic --checkpoint {final_path}")
    print("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train Neuro-Symbolic Agent"
    )

    parser.add_argument(
        '--demo-file',
        type=str,
        default='demonstrations.pkl',
        help='Demonstration file'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/neuro_symbolic',
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='hybrid',
        choices=['transformer', 'hierarchical', 'hybrid'],
        help='Agent mode'
    )

    parser.add_argument(
        '--no-transformer',
        action='store_true',
        help='Disable transformer attention'
    )

    parser.add_argument(
        '--no-hierarchical',
        action='store_true',
        help='Disable hierarchical policy'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device'
    )

    args = parser.parse_args()

    try:
        train_neuro_symbolic(
            demo_file=args.demo_file,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_transformer=not args.no_transformer,
            use_hierarchical=not args.no_hierarchical,
            mode=args.mode,
            device=args.device
        )
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
