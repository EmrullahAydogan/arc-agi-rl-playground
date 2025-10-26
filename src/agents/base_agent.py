"""
Base Agent Interface
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class BaseAgent(ABC):
    """Tüm agent'lar için abstract base class"""

    def __init__(self, action_space, observation_space):
        """
        Args:
            action_space: Gymnasium action space
            observation_space: Gymnasium observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def select_action(self, observation: np.ndarray, info: Dict = None) -> int:
        """
        Observation'a göre action seç

        Args:
            observation: Environment observation
            info: Ek bilgiler

        Returns:
            Seçilen action
        """
        pass

    @abstractmethod
    def update(self, observation: np.ndarray, action: int, reward: float,
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """
        Agent'ı güncelle (learning için)

        Args:
            observation: Önceki observation
            action: Alınan action
            reward: Alınan reward
            next_observation: Yeni observation
            done: Episode bitti mi?
            info: Ek bilgiler
        """
        pass

    def reset(self):
        """Episode başında agent'ı reset et"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Agent bilgilerini döndür"""
        return {
            'type': self.__class__.__name__
        }

    def get_params(self) -> Dict[str, Any]:
        """Agent parametrelerini döndür (override edilebilir)"""
        return {
            'learning_rate': 'N/A',
            'epsilon': 'N/A',
            'gamma': 'N/A',
            'batch_size': 'N/A',
            'memory_size': 'N/A',
            'update_frequency': 'N/A'
        }
