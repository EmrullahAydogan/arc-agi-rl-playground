"""
Random Agent - Baseline agent
"""
import numpy as np
from typing import Dict
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Rastgele action seçen basit agent"""

    def __init__(self, action_space, observation_space, seed: int = None):
        """
        Args:
            action_space: Gymnasium action space
            observation_space: Gymnasium observation space
            seed: Random seed
        """
        super().__init__(action_space, observation_space)
        self.rng = np.random.RandomState(seed)
        self.action_count = 0

    def select_action(self, observation: np.ndarray, info: Dict = None) -> int:
        """Rastgele bir action seç"""
        self.action_count += 1
        return self.action_space.sample()

    def update(self, observation: np.ndarray, action: int, reward: float,
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """Random agent öğrenmez, bu yüzden boş"""
        pass

    def reset(self):
        """Episode başında sayacı sıfırla"""
        self.action_count = 0

    def get_info(self) -> Dict:
        """Agent bilgilerini döndür"""
        return {
            'type': 'RandomAgent',
            'actions_taken': self.action_count
        }

    def get_params(self) -> Dict:
        """Agent parametrelerini döndür"""
        return {
            'learning_rate': 'N/A (Random)',
            'epsilon': '1.0 (Pure random)',
            'gamma': 'N/A',
            'batch_size': 'N/A',
            'memory_size': 'N/A',
            'update_frequency': 'N/A'
        }
