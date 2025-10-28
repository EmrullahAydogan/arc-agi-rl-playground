"""
Demonstration Buffer for Human Teaching
Stores human demonstrations for imitation learning / behavioral cloning
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json


class Demonstration:
    """Single demonstration (one puzzle solve attempt)"""

    def __init__(self, puzzle_id: str, mode: str = 'train', sample_idx: int = 0):
        """
        Args:
            puzzle_id: ID of the puzzle
            mode: 'train' or 'test'
            sample_idx: Training sample index (for train mode)
        """
        self.puzzle_id = puzzle_id
        self.mode = mode
        self.sample_idx = sample_idx

        # Episode data
        self.states = []  # List of grid states (numpy arrays)
        self.actions = []  # List of actions (int)
        self.rewards = []  # List of rewards (float)
        self.info_history = []  # List of info dicts

        # Metadata
        self.total_steps = 0
        self.total_reward = 0.0
        self.solved = False
        self.final_accuracy = 0.0

        # Timing
        self.start_time = None
        self.end_time = None
        self.duration_seconds = 0.0

    def add_step(self, state: np.ndarray, action: int, reward: float, info: dict):
        """Add a single step to the demonstration"""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.info_history.append(info.copy())

        self.total_steps += 1
        self.total_reward += reward

    def finalize(self, solved: bool, final_accuracy: float, duration: float):
        """Mark demonstration as complete"""
        self.solved = solved
        self.final_accuracy = final_accuracy
        self.duration_seconds = duration

    def get_state_action_pairs(self) -> List[Tuple[np.ndarray, int]]:
        """Get all (state, action) pairs for training"""
        return list(zip(self.states, self.actions))

    def get_successful_pairs(self, accuracy_threshold: float = 90.0) -> List[Tuple[np.ndarray, int]]:
        """
        Get only state-action pairs from successful demonstrations

        Args:
            accuracy_threshold: Minimum accuracy to consider successful

        Returns:
            List of (state, action) pairs
        """
        if self.final_accuracy >= accuracy_threshold or self.solved:
            return self.get_state_action_pairs()
        return []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'puzzle_id': self.puzzle_id,
            'mode': self.mode,
            'sample_idx': self.sample_idx,
            'states': [s.tolist() for s in self.states],
            'actions': self.actions,
            'rewards': self.rewards,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'solved': self.solved,
            'final_accuracy': self.final_accuracy,
            'duration_seconds': self.duration_seconds
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Demonstration':
        """Create demonstration from dictionary"""
        demo = cls(
            puzzle_id=data['puzzle_id'],
            mode=data['mode'],
            sample_idx=data['sample_idx']
        )

        demo.states = [np.array(s) for s in data['states']]
        demo.actions = data['actions']
        demo.rewards = data['rewards']
        demo.total_steps = data['total_steps']
        demo.total_reward = data['total_reward']
        demo.solved = data['solved']
        demo.final_accuracy = data['final_accuracy']
        demo.duration_seconds = data['duration_seconds']

        return demo


class DemonstrationBuffer:
    """
    Buffer for storing human demonstrations
    Supports saving/loading, filtering, and sampling for imitation learning
    """

    def __init__(self, save_dir: str = "demonstrations"):
        """
        Args:
            save_dir: Directory to save demonstrations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.demonstrations: List[Demonstration] = []

        # Index by puzzle_id for quick access
        self.demos_by_puzzle: Dict[str, List[Demonstration]] = defaultdict(list)

        # Statistics
        self.total_demonstrations = 0
        self.total_steps = 0
        self.solved_count = 0

    def add_demonstration(self, demo: Demonstration):
        """Add a demonstration to the buffer"""
        self.demonstrations.append(demo)
        self.demos_by_puzzle[demo.puzzle_id].append(demo)

        # Update stats
        self.total_demonstrations += 1
        self.total_steps += demo.total_steps
        if demo.solved:
            self.solved_count += 1

        print(f"\n[DEMO BUFFER] Added demonstration:")
        print(f"  Puzzle: {demo.puzzle_id}")
        print(f"  Steps: {demo.total_steps}")
        print(f"  Reward: {demo.total_reward:.2f}")
        print(f"  Solved: {demo.solved}")
        print(f"  Accuracy: {demo.final_accuracy:.1f}%")
        print(f"  Duration: {demo.duration_seconds:.1f}s")
        print(f"  Total demos: {self.total_demonstrations}")

    def get_all_state_action_pairs(self, only_successful: bool = False,
                                   accuracy_threshold: float = 90.0) -> List[Tuple[np.ndarray, int]]:
        """
        Get all state-action pairs from demonstrations

        Args:
            only_successful: Only include successful demonstrations
            accuracy_threshold: Minimum accuracy for successful demos

        Returns:
            List of (state, action) pairs
        """
        pairs = []

        for demo in self.demonstrations:
            if only_successful:
                pairs.extend(demo.get_successful_pairs(accuracy_threshold))
            else:
                pairs.extend(demo.get_state_action_pairs())

        return pairs

    def get_demonstrations_for_puzzle(self, puzzle_id: str) -> List[Demonstration]:
        """Get all demonstrations for a specific puzzle"""
        return self.demos_by_puzzle[puzzle_id]

    def get_best_demonstration(self, puzzle_id: str) -> Optional[Demonstration]:
        """Get the best demonstration for a puzzle (highest accuracy)"""
        demos = self.demos_by_puzzle[puzzle_id]
        if not demos:
            return None

        # Sort by: solved (yes/no), then by accuracy, then by fewest steps
        return max(demos, key=lambda d: (d.solved, d.final_accuracy, -d.total_steps))

    def sample_batch(self, batch_size: int, only_successful: bool = True,
                    accuracy_threshold: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of (state, action) pairs for training

        Args:
            batch_size: Number of samples to return
            only_successful: Only sample from successful demonstrations
            accuracy_threshold: Minimum accuracy for successful demos

        Returns:
            states: (batch_size, H, W) numpy array
            actions: (batch_size,) numpy array
        """
        pairs = self.get_all_state_action_pairs(only_successful, accuracy_threshold)

        if len(pairs) == 0:
            raise ValueError("No demonstrations available for sampling!")

        # Random sample with replacement
        indices = np.random.randint(0, len(pairs), size=batch_size)

        sampled_pairs = [pairs[i] for i in indices]
        states = np.array([s for s, a in sampled_pairs])
        actions = np.array([a for s, a in sampled_pairs])

        return states, actions

    def save(self, filename: str = "demonstrations.pkl"):
        """Save all demonstrations to file"""
        filepath = self.save_dir / filename

        # Convert to serializable format
        data = {
            'demonstrations': [demo.to_dict() for demo in self.demonstrations],
            'total_demonstrations': self.total_demonstrations,
            'total_steps': self.total_steps,
            'solved_count': self.solved_count
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n[DEMO BUFFER] Saved {self.total_demonstrations} demonstrations to {filepath}")

        # Also save human-readable JSON summary
        summary_path = self.save_dir / filename.replace('.pkl', '_summary.json')
        summary = self.get_summary()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[DEMO BUFFER] Saved summary to {summary_path}")

    def load(self, filename: str = "demonstrations.pkl"):
        """Load demonstrations from file"""
        filepath = self.save_dir / filename

        if not filepath.exists():
            print(f"[DEMO BUFFER] No saved demonstrations found at {filepath}")
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Restore demonstrations
        self.demonstrations = [Demonstration.from_dict(d) for d in data['demonstrations']]
        self.total_demonstrations = data['total_demonstrations']
        self.total_steps = data['total_steps']
        self.solved_count = data['solved_count']

        # Rebuild index
        self.demos_by_puzzle.clear()
        for demo in self.demonstrations:
            self.demos_by_puzzle[demo.puzzle_id].append(demo)

        print(f"\n[DEMO BUFFER] Loaded {self.total_demonstrations} demonstrations from {filepath}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Solved: {self.solved_count}/{self.total_demonstrations}")

    def get_summary(self) -> dict:
        """Get summary statistics"""
        if self.total_demonstrations == 0:
            return {
                'total_demonstrations': 0,
                'total_steps': 0,
                'solved_count': 0,
                'solve_rate': 0.0,
                'avg_steps': 0.0,
                'avg_reward': 0.0,
                'puzzles_covered': 0
            }

        avg_steps = self.total_steps / self.total_demonstrations
        avg_reward = sum(d.total_reward for d in self.demonstrations) / self.total_demonstrations
        solve_rate = (self.solved_count / self.total_demonstrations) * 100

        # Puzzle coverage
        puzzles_covered = len(self.demos_by_puzzle)

        # Per-puzzle stats
        puzzle_stats = {}
        for puzzle_id, demos in self.demos_by_puzzle.items():
            best_demo = self.get_best_demonstration(puzzle_id)
            puzzle_stats[puzzle_id] = {
                'num_attempts': len(demos),
                'solved': any(d.solved for d in demos),
                'best_accuracy': best_demo.final_accuracy if best_demo else 0.0,
                'avg_steps': sum(d.total_steps for d in demos) / len(demos)
            }

        return {
            'total_demonstrations': self.total_demonstrations,
            'total_steps': self.total_steps,
            'solved_count': self.solved_count,
            'solve_rate': solve_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'puzzles_covered': puzzles_covered,
            'puzzle_stats': puzzle_stats
        }

    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("DEMONSTRATION BUFFER SUMMARY")
        print("="*60)
        print(f"Total Demonstrations: {summary['total_demonstrations']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Solved: {summary['solved_count']}/{summary['total_demonstrations']} ({summary['solve_rate']:.1f}%)")
        print(f"Avg Steps per Demo: {summary['avg_steps']:.1f}")
        print(f"Avg Reward per Demo: {summary['avg_reward']:.2f}")
        print(f"Puzzles Covered: {summary['puzzles_covered']}")
        print("="*60)

        # Top 5 puzzles by attempts
        if 'puzzle_stats' in summary and summary['puzzle_stats']:
            print("\nTop Puzzles by Attempts:")
            sorted_puzzles = sorted(
                summary['puzzle_stats'].items(),
                key=lambda x: x[1]['num_attempts'],
                reverse=True
            )[:5]

            for puzzle_id, stats in sorted_puzzles:
                solved_mark = "✓" if stats['solved'] else "✗"
                print(f"  {solved_mark} {puzzle_id[:20]:20s} - {stats['num_attempts']} attempts, "
                      f"{stats['best_accuracy']:.1f}% best accuracy")

        print()

    def clear(self):
        """Clear all demonstrations"""
        self.demonstrations.clear()
        self.demos_by_puzzle.clear()
        self.total_demonstrations = 0
        self.total_steps = 0
        self.solved_count = 0
        print("[DEMO BUFFER] Cleared all demonstrations")
