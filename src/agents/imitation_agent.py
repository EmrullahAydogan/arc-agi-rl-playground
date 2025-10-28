"""
Imitation Learning Agent (Behavioral Cloning)
Learns from human demonstrations to solve ARC puzzles
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.utils.demonstration_buffer import DemonstrationBuffer


class CNNPolicy(nn.Module):
    """
    CNN-based policy network for ARC puzzles
    Input: (H, W) grid with values 0-9
    Output: (30*30*10 + 4) action probabilities
    """

    def __init__(self, max_grid_size: int = 30, num_colors: int = 10, num_resize_actions: int = 4):
        super(CNNPolicy, self).__init__()

        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.num_paint_actions = max_grid_size * max_grid_size * num_colors
        self.num_resize_actions = num_resize_actions
        self.total_actions = self.num_paint_actions + num_resize_actions

        # Input: (1, max_grid_size, max_grid_size) - single channel with color IDs
        # We'll embed each color into a higher dimensional space

        # CNN layers
        self.conv1 = nn.Conv2d(num_colors, 64, kernel_size=3, padding=1)  # (64, H, W)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (128, H, W)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (256, H, W)
        self.bn3 = nn.BatchNorm2d(256)

        # Global features
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (256, 1, 1)

        # Action heads
        # Paint action head: (x, y, color) spatial predictions
        self.paint_conv = nn.Conv2d(256, num_colors, kernel_size=1)  # (num_colors, H, W)

        # Resize action head: simple 4-way classification
        self.resize_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_resize_actions)
        )

        # Action type head: paint vs resize
        self.action_type_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [paint, resize]
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: (batch, max_grid_size, max_grid_size) with values 0-9

        Returns:
            action_type_logits: (batch, 2) - [paint, resize]
            paint_logits: (batch, num_colors, H, W) - spatial paint predictions
            resize_logits: (batch, 4) - resize action predictions
        """
        batch_size = state.shape[0]

        # One-hot encode colors: (batch, num_colors, H, W)
        state_onehot = F.one_hot(state.long(), num_classes=self.num_colors).permute(0, 3, 1, 2).float()

        # CNN layers
        x = F.relu(self.bn1(self.conv1(state_onehot)))  # (batch, 64, H, W)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, H, W)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 256, H, W)

        # Global features for classification heads
        global_features = self.global_pool(x).view(batch_size, 256)  # (batch, 256)

        # Action type prediction
        action_type_logits = self.action_type_fc(global_features)  # (batch, 2)

        # Paint action prediction (spatial)
        paint_logits = self.paint_conv(x)  # (batch, num_colors, H, W)

        # Resize action prediction
        resize_logits = self.resize_fc(global_features)  # (batch, 4)

        return action_type_logits, paint_logits, resize_logits

    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get full action logits for all actions

        Args:
            state: (batch, H, W) grid

        Returns:
            logits: (batch, total_actions) - logits for all actions
        """
        action_type_logits, paint_logits, resize_logits = self.forward(state)

        batch_size = state.shape[0]

        # Flatten paint logits: (batch, num_colors * H * W)
        paint_logits_flat = paint_logits.view(batch_size, -1)

        # Combine: action_type determines which logits to use
        # Simple approach: concatenate and let softmax handle it
        # We'll weight the logits by action type probability

        # Get action type probabilities
        action_type_probs = F.softmax(action_type_logits, dim=1)  # (batch, 2)
        paint_prob = action_type_probs[:, 0:1]  # (batch, 1)
        resize_prob = action_type_probs[:, 1:2]  # (batch, 1)

        # Weight logits
        paint_logits_weighted = paint_logits_flat + torch.log(paint_prob + 1e-8)
        resize_logits_weighted = resize_logits + torch.log(resize_prob + 1e-8)

        # Concatenate: [paint_actions, resize_actions]
        all_logits = torch.cat([paint_logits_weighted, resize_logits_weighted], dim=1)

        return all_logits


class ImitationAgent(BaseAgent):
    """
    Imitation Learning Agent using Behavioral Cloning
    Learns to mimic human demonstrations
    """

    def __init__(
        self,
        action_space_size: int,
        max_grid_size: int = 30,
        num_colors: int = 10,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints/imitation"
    ):
        """
        Args:
            action_space_size: Total number of actions (9004)
            max_grid_size: Maximum grid dimension (30)
            num_colors: Number of ARC colors (10)
            learning_rate: Learning rate for optimizer
            device: "cuda" or "cpu"
            checkpoint_dir: Directory to save checkpoints
        """
        super().__init__()

        self.action_space_size = action_space_size
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.device = torch.device(device)

        # Policy network
        self.policy = CNNPolicy(
            max_grid_size=max_grid_size,
            num_colors=num_colors,
            num_resize_actions=4
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Training stats
        self.learning_rate = learning_rate
        self.training_steps = 0
        self.total_loss = 0.0

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Exploration
        self.epsilon = 0.0  # No exploration during evaluation (use human demos only)

        print(f"\n[IMITATION AGENT] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Action space: {action_space_size}")
        print(f"  Learning rate: {learning_rate}")

    def select_action(self, state: np.ndarray, mode: str = "exploit") -> int:
        """
        Select action given current state

        Args:
            state: (H, W) grid with values 0-9
            mode: "exploit" (use learned policy) or "explore" (add noise)

        Returns:
            action: Integer action ID
        """
        # Convert to tensor and add batch dimension
        state_tensor = self._prepare_state(state)

        with torch.no_grad():
            # Get action logits
            logits = self.policy.get_action_logits(state_tensor)  # (1, action_space_size)

            # Select action
            if mode == "explore" and np.random.rand() < self.epsilon:
                # Random action
                action = np.random.randint(0, self.action_space_size)
            else:
                # Greedy action
                action = torch.argmax(logits, dim=1).item()

        return action

    def update(self, demonstration_buffer: DemonstrationBuffer, batch_size: int = 32,
              num_epochs: int = 1, only_successful: bool = True) -> Dict[str, float]:
        """
        Update policy using demonstrations from buffer

        Args:
            demonstration_buffer: Buffer containing human demonstrations
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            only_successful: Only train on successful demonstrations

        Returns:
            metrics: Dictionary with training metrics
        """
        self.policy.train()

        epoch_losses = []

        for epoch in range(num_epochs):
            # Sample batch from demonstration buffer
            try:
                states, actions = demonstration_buffer.sample_batch(
                    batch_size=batch_size,
                    only_successful=only_successful,
                    accuracy_threshold=90.0
                )
            except ValueError as e:
                print(f"[IMITATION AGENT] {e}")
                return {'loss': 0.0, 'accuracy': 0.0}

            # Prepare tensors
            states_tensor = self._prepare_state(states)  # (batch, H, W)
            actions_tensor = torch.LongTensor(actions).to(self.device)  # (batch,)

            # Forward pass
            logits = self.policy.get_action_logits(states_tensor)  # (batch, action_space_size)

            # Cross-entropy loss
            loss = F.cross_entropy(logits, actions_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Stats
            epoch_losses.append(loss.item())
            self.training_steps += 1
            self.total_loss += loss.item()

            # Accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == actions_tensor).float().mean().item()

        avg_loss = np.mean(epoch_losses)

        return {
            'loss': avg_loss,
            'accuracy': accuracy * 100,
            'training_steps': self.training_steps
        }

    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Prepare state for network

        Args:
            state: (H, W) or (batch, H, W) numpy array

        Returns:
            tensor: (batch, max_grid_size, max_grid_size) tensor
        """
        if state.ndim == 2:
            state = state[np.newaxis, ...]  # Add batch dimension

        batch_size = state.shape[0]
        H, W = state.shape[1], state.shape[2]

        # Pad to max_grid_size
        padded = np.zeros((batch_size, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        padded[:, :H, :W] = state

        # Convert to tensor
        tensor = torch.FloatTensor(padded).to(self.device)

        return tensor

    def save(self, filepath: str):
        """Save agent checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'total_loss': self.total_loss,
            'learning_rate': self.learning_rate
        }

        torch.save(checkpoint, filepath)
        print(f"[IMITATION AGENT] Saved checkpoint to {filepath}")

    def load(self, filepath: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.total_loss = checkpoint['total_loss']

        print(f"[IMITATION AGENT] Loaded checkpoint from {filepath}")
        print(f"  Training steps: {self.training_steps}")

    def get_params(self) -> Dict:
        """Get agent parameters for visualization"""
        return {
            'agent_type': 'ImitationLearning',
            'learning_rate': self.learning_rate,
            'training_steps': self.training_steps,
            'avg_loss': self.total_loss / max(1, self.training_steps),
            'device': str(self.device),
            'epsilon': self.epsilon
        }
