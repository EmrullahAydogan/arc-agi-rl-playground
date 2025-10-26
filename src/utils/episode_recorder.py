#!/usr/bin/env python3
"""
Episode Recorder - Records episode history with detailed metrics
"""
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path


class Episode:
    """Single episode record"""

    def __init__(
        self,
        episode_id: int,
        puzzle_id: str,
        dataset: str,
        mode: str,
        sample_index: int = 0
    ):
        """
        Args:
            episode_id: Unique episode identifier
            puzzle_id: Puzzle identifier
            dataset: "training" or "evaluation"
            mode: "train" or "test"
            sample_index: Sample index (for train mode)
        """
        self.episode_id = episode_id
        self.puzzle_id = puzzle_id
        self.dataset = dataset
        self.mode = mode
        self.sample_index = sample_index

        # Episode metrics
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = 0.0  # seconds

        self.total_steps = 0
        self.total_reward = 0.0
        self.final_accuracy = 0.0
        self.is_solved = False

        # Step-by-step recording for replay
        self.actions = []  # List of actions taken
        self.rewards = []  # Reward for each action
        self.accuracies = []  # Accuracy after each action
        self.grids = []  # Grid state after each action

    def record_step(
        self,
        action: int,
        reward: float,
        accuracy: float,
        grid_state: Optional[list] = None
    ):
        """Record a single step"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.accuracies.append(accuracy)
        if grid_state is not None:
            self.grids.append(grid_state)
        self.total_steps += 1

    def end_episode(self, is_solved: bool, final_accuracy: float):
        """Mark episode as complete"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.is_solved = is_solved
        self.final_accuracy = final_accuracy
        self.total_reward = sum(self.rewards)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'episode_id': self.episode_id,
            'puzzle_id': self.puzzle_id,
            'dataset': self.dataset,
            'mode': self.mode,
            'sample_index': self.sample_index,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'final_accuracy': self.final_accuracy,
            'is_solved': self.is_solved,
            'actions': self.actions,
            'rewards': self.rewards,
            'accuracies': self.accuracies,
            # Don't save grids to save space
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Episode':
        """Create Episode from dictionary"""
        episode = cls(
            episode_id=data['episode_id'],
            puzzle_id=data['puzzle_id'],
            dataset=data['dataset'],
            mode=data['mode'],
            sample_index=data['sample_index']
        )
        episode.start_time = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            episode.end_time = datetime.fromisoformat(data['end_time'])
        episode.duration = data['duration']
        episode.total_steps = data['total_steps']
        episode.total_reward = data['total_reward']
        episode.final_accuracy = data['final_accuracy']
        episode.is_solved = data['is_solved']
        episode.actions = data.get('actions', [])
        episode.rewards = data.get('rewards', [])
        episode.accuracies = data.get('accuracies', [])
        return episode


class EpisodeRecorder:
    """Records and manages episode history"""

    def __init__(self, max_episodes: int = 1000):
        """
        Args:
            max_episodes: Maximum number of episodes to keep in memory
        """
        self.max_episodes = max_episodes
        self.episodes: List[Episode] = []
        self.current_episode: Optional[Episode] = None
        self.next_episode_id = 1

    def start_episode(
        self,
        puzzle_id: str,
        dataset: str,
        mode: str,
        sample_index: int = 0
    ) -> Episode:
        """Start recording a new episode"""
        episode = Episode(
            episode_id=self.next_episode_id,
            puzzle_id=puzzle_id,
            dataset=dataset,
            mode=mode,
            sample_index=sample_index
        )
        self.current_episode = episode
        self.next_episode_id += 1
        return episode

    def record_step(
        self,
        action: int,
        reward: float,
        accuracy: float,
        grid_state: Optional[list] = None
    ):
        """Record a step in the current episode"""
        if self.current_episode:
            self.current_episode.record_step(action, reward, accuracy, grid_state)

    def end_episode(self, is_solved: bool, final_accuracy: float):
        """End the current episode and save it"""
        if self.current_episode:
            self.current_episode.end_episode(is_solved, final_accuracy)
            self.episodes.append(self.current_episode)

            # Keep only max_episodes most recent
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)

            self.current_episode = None

    def get_episodes(
        self,
        puzzle_id: Optional[str] = None,
        mode: Optional[str] = None,
        solved_only: bool = False
    ) -> List[Episode]:
        """
        Get episodes with optional filtering

        Args:
            puzzle_id: Filter by puzzle ID
            mode: Filter by mode ("train" or "test")
            solved_only: Only return solved episodes

        Returns:
            List of matching episodes
        """
        episodes = self.episodes

        if puzzle_id:
            episodes = [e for e in episodes if e.puzzle_id == puzzle_id]

        if mode:
            episodes = [e for e in episodes if e.mode == mode]

        if solved_only:
            episodes = [e for e in episodes if e.is_solved]

        return episodes

    def get_episode_by_id(self, episode_id: int) -> Optional[Episode]:
        """Get a specific episode by ID"""
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def get_statistics(self) -> dict:
        """Get overall statistics"""
        if not self.episodes:
            return {
                'total_episodes': 0,
                'solved_episodes': 0,
                'success_rate': 0.0,
                'avg_steps': 0.0,
                'avg_reward': 0.0,
                'avg_accuracy': 0.0
            }

        solved = [e for e in self.episodes if e.is_solved]

        return {
            'total_episodes': len(self.episodes),
            'solved_episodes': len(solved),
            'success_rate': len(solved) / len(self.episodes) * 100,
            'avg_steps': sum(e.total_steps for e in self.episodes) / len(self.episodes),
            'avg_reward': sum(e.total_reward for e in self.episodes) / len(self.episodes),
            'avg_accuracy': sum(e.final_accuracy for e in self.episodes) / len(self.episodes) * 100
        }

    def clear(self):
        """Clear all episodes"""
        self.episodes.clear()
        self.current_episode = None

    def save_to_file(self, filepath: str):
        """Save episode history to JSON file"""
        data = {
            'next_episode_id': self.next_episode_id,
            'episodes': [e.to_dict() for e in self.episodes]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load episode history from JSON file"""
        if not Path(filepath).exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.next_episode_id = data.get('next_episode_id', 1)
        self.episodes = [Episode.from_dict(e) for e in data.get('episodes', [])]


if __name__ == "__main__":
    # Test the recorder
    recorder = EpisodeRecorder()

    # Record a test episode
    episode = recorder.start_episode("test_puzzle", "training", "train", 0)
    recorder.record_step(0, 0.5, 0.3)
    recorder.record_step(1, 0.7, 0.5)
    recorder.record_step(2, 1.0, 0.8)
    recorder.end_episode(True, 0.8)

    print("Episode recorded!")
    print(f"Total episodes: {len(recorder.episodes)}")
    print(f"Statistics: {recorder.get_statistics()}")
