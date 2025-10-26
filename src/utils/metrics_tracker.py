"""
Performance Metrics Tracker
Train ve Test modları için performans metriklerini takip eder
"""
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


class MetricsTracker:
    """
    Agent'ın train ve test performansını takip eder

    Train Mode Metrics:
    - Episode sayısı
    - Çözülen puzzle sayısı
    - Ortalama reward
    - Ortalama accuracy
    - Sample bazında performans

    Test Mode Metrics:
    - Test edilen puzzle sayısı
    - Başarı oranı
    - En iyi accuracy
    - Ortalama accuracy
    """

    def __init__(self):
        """MetricsTracker'ı başlat"""
        # Training metrics
        self.train_episodes = 0
        self.train_solved = 0
        self.train_rewards = []
        self.train_accuracies = []
        self.train_steps = []

        # Sample bazında train metrics
        self.train_sample_stats = defaultdict(lambda: {
            'episodes': 0,
            'solved': 0,
            'rewards': [],
            'accuracies': [],
            'steps': []
        })

        # Test metrics
        self.test_episodes = 0
        self.test_solved = 0
        self.test_accuracies = []
        self.test_steps = []

        # Current episode tracking
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_mode = 'train'
        self.current_sample_index = 0

    def start_episode(self, mode: str, sample_index: int = 0):
        """
        Yeni episode başlat

        Args:
            mode: 'train' veya 'test'
            sample_index: Train sample index (train mode için)
        """
        self.current_mode = mode
        self.current_sample_index = sample_index
        self.current_episode_rewards = []
        self.current_episode_steps = 0

    def record_step(self, reward: float, accuracy: float):
        """
        Bir adımı kaydet

        Args:
            reward: Bu adımdan alınan reward
            accuracy: Mevcut accuracy (0-1 arası)
        """
        self.current_episode_rewards.append(reward)
        self.current_episode_steps += 1

    def end_episode(self, is_solved: bool, final_accuracy: float):
        """
        Episode'u sonlandır ve metrikleri kaydet

        Args:
            is_solved: Puzzle çözüldü mü?
            final_accuracy: Final accuracy (0-1 arası)
        """
        total_reward = sum(self.current_episode_rewards)

        if self.current_mode == 'train':
            # Global train metrics
            self.train_episodes += 1
            if is_solved:
                self.train_solved += 1
            self.train_rewards.append(total_reward)
            self.train_accuracies.append(final_accuracy)
            self.train_steps.append(self.current_episode_steps)

            # Sample-specific metrics
            sample_stats = self.train_sample_stats[self.current_sample_index]
            sample_stats['episodes'] += 1
            if is_solved:
                sample_stats['solved'] += 1
            sample_stats['rewards'].append(total_reward)
            sample_stats['accuracies'].append(final_accuracy)
            sample_stats['steps'].append(self.current_episode_steps)

        else:  # test mode
            self.test_episodes += 1
            if is_solved:
                self.test_solved += 1
            self.test_accuracies.append(final_accuracy)
            self.test_steps.append(self.current_episode_steps)

    def get_train_metrics(self) -> Dict:
        """
        Train metrikleri döndür

        Returns:
            Train metrics dict
        """
        if self.train_episodes == 0:
            return {
                'episodes': 0,
                'solved': 0,
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_accuracy': 0.0,
                'avg_steps': 0.0,
                'best_accuracy': 0.0
            }

        return {
            'episodes': self.train_episodes,
            'solved': self.train_solved,
            'success_rate': self.train_solved / self.train_episodes * 100,
            'avg_reward': np.mean(self.train_rewards),
            'avg_accuracy': np.mean(self.train_accuracies) * 100,
            'avg_steps': np.mean(self.train_steps),
            'best_accuracy': max(self.train_accuracies) * 100 if self.train_accuracies else 0.0
        }

    def get_test_metrics(self) -> Dict:
        """
        Test metrikleri döndür

        Returns:
            Test metrics dict
        """
        if self.test_episodes == 0:
            return {
                'episodes': 0,
                'solved': 0,
                'success_rate': 0.0,
                'avg_accuracy': 0.0,
                'avg_steps': 0.0,
                'best_accuracy': 0.0
            }

        return {
            'episodes': self.test_episodes,
            'solved': self.test_solved,
            'success_rate': self.test_solved / self.test_episodes * 100,
            'avg_accuracy': np.mean(self.test_accuracies) * 100,
            'avg_steps': np.mean(self.test_steps),
            'best_accuracy': max(self.test_accuracies) * 100 if self.test_accuracies else 0.0
        }

    def get_sample_metrics(self, sample_index: int) -> Dict:
        """
        Belirli bir train sample için metrikleri döndür

        Args:
            sample_index: Sample index

        Returns:
            Sample metrics dict
        """
        if sample_index not in self.train_sample_stats:
            return {
                'episodes': 0,
                'solved': 0,
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_accuracy': 0.0
            }

        stats = self.train_sample_stats[sample_index]

        if stats['episodes'] == 0:
            return {
                'episodes': 0,
                'solved': 0,
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_accuracy': 0.0
            }

        return {
            'episodes': stats['episodes'],
            'solved': stats['solved'],
            'success_rate': stats['solved'] / stats['episodes'] * 100,
            'avg_reward': np.mean(stats['rewards']),
            'avg_accuracy': np.mean(stats['accuracies']) * 100
        }

    def get_all_metrics(self) -> Dict:
        """
        Tüm metrikleri döndür

        Returns:
            Tüm metrics dict
        """
        # Tüm train sample'ların metrics'lerini hazırla
        train_sample_stats = {}
        for sample_idx in self.train_sample_stats.keys():
            train_sample_stats[sample_idx] = self.get_sample_metrics(sample_idx)

        return {
            'train': self.get_train_metrics(),
            'test': self.get_test_metrics(),
            'current_sample': self.get_sample_metrics(self.current_sample_index),
            'train_sample_stats': train_sample_stats
        }

    def reset(self):
        """Tüm metrikleri sıfırla"""
        self.__init__()
