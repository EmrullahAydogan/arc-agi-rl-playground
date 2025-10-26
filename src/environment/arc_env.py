"""
ARC-AGI Reinforcement Learning Environment
Gymnasium (OpenAI Gym) tarzı environment
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import copy


class ARCEnvironment(gym.Env):
    """
    ARC puzzle'ları için RL environment

    Action Space: Discrete actions representing (x, y, color) combinations
    Observation Space: Grid as numpy array
    """

    def __init__(
        self,
        puzzle_data: Dict,
        task_index: int = 0,
        max_steps: int = 100,
        max_grid_size: int = 30,
        num_colors: int = 10,
        mode: str = 'train'
    ):
        """
        Args:
            puzzle_data: ARC puzzle verisi (train ve test içeren)
            task_index: Hangi test görevini kullanacağız (default: 0)
            max_steps: Maksimum adım sayısı
            max_grid_size: Maksimum grid boyutu
            num_colors: Renk sayısı (ARC'de 0-9 arası)
            mode: 'train' veya 'test' - başlangıç modu
        """
        super().__init__()

        self.puzzle_data = puzzle_data
        self.task_index = task_index
        self.max_steps = max_steps
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors

        # Mode sistemi
        self.mode = mode  # 'train' veya 'test'
        self.train_sample_index = 0  # Train örnekleri arasında gezinmek için
        self.num_train_samples = len(puzzle_data.get('train', []))
        self.num_test_samples = len(puzzle_data.get('test', []))

        # Mevcut task'ı yükle
        self._load_current_task()

        # Grid boyutları
        self.grid_height, self.grid_width = self.input_grid.shape

        # Action space: Her (x, y, color) kombinasyonu bir action
        # Basitleştirme için discrete action space kullanıyoruz
        self.action_space = spaces.Discrete(
            self.max_grid_size * self.max_grid_size * self.num_colors
        )

        # Observation space: Grid (max_grid_size x max_grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_colors - 1,
            shape=(self.max_grid_size, self.max_grid_size),
            dtype=np.int32
        )

        # State
        self.current_grid = None
        self.steps = 0
        self.done = False

        # Metrics
        self.total_reward = 0
        self.last_action = None
        self.last_reward = 0

    def _load_current_task(self):
        """Mevcut mode ve sample index'e göre task'ı yükle"""
        if self.mode == 'train':
            # Train mode: Train örneklerinden birini kullan
            if self.num_train_samples == 0:
                raise ValueError("No training samples available!")

            train_task = self.puzzle_data['train'][self.train_sample_index]
            self.input_grid = np.array(train_task['input'])
            self.target_grid = np.array(train_task['output'])

        else:  # test mode
            # Test mode: Test görevini kullan
            if self.num_test_samples == 0:
                raise ValueError("No test samples available!")

            test_task = self.puzzle_data['test'][self.task_index]
            self.input_grid = np.array(test_task['input'])

            # Test'te output olmayabilir
            if 'output' in test_task:
                self.target_grid = np.array(test_task['output'])
            else:
                # Output yoksa, boş bir grid oluştur (agent'ın tahmin etmesi için)
                self.target_grid = np.zeros_like(self.input_grid)

    def switch_mode(self) -> str:
        """Train ve Test modları arasında geçiş yap

        Returns:
            Yeni mod ('train' veya 'test')
        """
        if self.mode == 'train':
            self.mode = 'test'
        else:
            self.mode = 'train'

        # Yeni task'ı yükle
        self._load_current_task()
        self.grid_height, self.grid_width = self.input_grid.shape

        return self.mode

    def next_sample(self) -> int:
        """Sonraki sample'a geç (sadece train mode'da)

        Returns:
            Yeni sample index
        """
        if self.mode == 'train' and self.num_train_samples > 1:
            self.train_sample_index = (self.train_sample_index + 1) % self.num_train_samples
            self._load_current_task()
            self.grid_height, self.grid_width = self.input_grid.shape

        return self.train_sample_index

    def previous_sample(self) -> int:
        """Önceki sample'a geç (sadece train mode'da)

        Returns:
            Yeni sample index
        """
        if self.mode == 'train' and self.num_train_samples > 1:
            self.train_sample_index = (self.train_sample_index - 1) % self.num_train_samples
            self._load_current_task()
            self.grid_height, self.grid_width = self.input_grid.shape

        return self.train_sample_index

    def _action_to_grid_operation(self, action: int) -> Tuple[int, int, int]:
        """
        Discrete action'ı (x, y, color) tuple'ına çevir

        Args:
            action: Discrete action ID

        Returns:
            (x, y, color) tuple
        """
        color = action % self.num_colors
        action //= self.num_colors
        y = action % self.max_grid_size
        x = action // self.max_grid_size
        return x, y, color

    def _compute_reward(self) -> float:
        """
        Current grid ile target grid arasındaki benzerliğe göre reward hesapla

        Returns:
            Reward değeri
        """
        # Target grid boyutlarını al
        target_height, target_width = self.target_grid.shape

        # Current grid'den target boyutunda slice al
        # Ensure we don't exceed current_grid dimensions
        max_h = min(target_height, self.current_grid.shape[0])
        max_w = min(target_width, self.current_grid.shape[1])
        current_slice = self.current_grid[:max_h, :max_w]

        # Target grid'den de aynı boyutta slice al (eğer boyutlar farklıysa)
        target_slice = self.target_grid[:max_h, :max_w]

        # Doğru piksel sayısı
        correct_pixels = np.sum(current_slice == target_slice)
        total_pixels = target_height * target_width

        # Accuracy-based reward
        accuracy = correct_pixels / total_pixels

        # Sparse reward: Tam doğruysa büyük ödül
        if accuracy == 1.0:
            return 100.0

        # Dense reward: İyileşme için küçük ödüller
        return accuracy * 10.0

    def _get_observation(self) -> np.ndarray:
        """Current grid'i observation olarak döndür"""
        return self.current_grid.copy()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Environment'ı reset et

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)

        # Grid'i input ile başlat
        self.current_grid = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        self.current_grid[:self.grid_height, :self.grid_width] = self.input_grid.copy()

        self.steps = 0
        self.done = False
        self.total_reward = 0
        self.last_action = None
        self.last_reward = 0

        info = {
            'puzzle_shape': (self.grid_height, self.grid_width),
            'target_grid': self.target_grid,
            'input_grid': self.input_grid
        }

        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Bir adım at

        Args:
            action: Alınacak aksiyon

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.steps += 1
        self.last_action = action

        # Action'ı uygula
        x, y, color = self._action_to_grid_operation(action)

        # Grid sınırları içinde mi kontrol et
        if x < self.grid_height and y < self.grid_width:
            self.current_grid[x, y] = color

        # Reward hesapla
        reward = self._compute_reward()
        self.last_reward = reward
        self.total_reward += reward

        # Episode bitti mi?
        terminated = self._is_solved()
        truncated = self.steps >= self.max_steps

        self.done = terminated or truncated

        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'action_decoded': (x, y, color),
            'is_solved': terminated
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _is_solved(self) -> bool:
        """Puzzle çözüldü mü?"""
        target_height, target_width = self.target_grid.shape

        # Ensure we don't exceed current_grid dimensions
        max_h = min(target_height, self.current_grid.shape[0])
        max_w = min(target_width, self.current_grid.shape[1])
        current_slice = self.current_grid[:max_h, :max_w]
        target_slice = self.target_grid[:max_h, :max_w]

        # If dimensions don't match exactly, puzzle isn't solved
        if max_h != target_height or max_w != target_width:
            return False

        return np.array_equal(current_slice, target_slice)

    def render(self) -> np.ndarray:
        """
        Grid'i render et (görselleştirme için)

        Returns:
            Current grid'in geçerli kısmı
        """
        return self.current_grid[:self.grid_height, :self.grid_width].copy()

    def get_state_info(self) -> Dict[str, Any]:
        """Environment state bilgilerini al"""
        return {
            'current_grid': self.render(),
            'target_grid': self.target_grid,
            'input_grid': self.input_grid,
            'steps': self.steps,
            'max_steps': self.max_steps,
            'total_reward': self.total_reward,
            'last_action': self.last_action,
            'last_reward': self.last_reward,
            'is_solved': self._is_solved(),
            'done': self.done,
            'mode': self.mode,
            'train_sample_index': self.train_sample_index,
            'num_train_samples': self.num_train_samples,
            'num_test_samples': self.num_test_samples
        }
