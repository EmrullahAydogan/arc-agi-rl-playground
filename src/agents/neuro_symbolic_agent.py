"""
Neuro-Symbolic Agent for ARC Puzzles
Combines neural perception with symbolic reasoning
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.perception.object_detector import ObjectDetector, ARCObject, extract_grid_features
from src.perception.attention_module import TransformerAgent
from src.symbolic.arc_dsl import ARCDSL, DSLObject
from src.policy.hierarchical_policy import HierarchicalPolicy, HighLevelOperation
from src.utils.demonstration_buffer import DemonstrationBuffer


class NeuroSymbolicAgent(BaseAgent):
    """
    Complete Neuro-Symbolic Agent

    Architecture:
    1. Perception Layer (Neural):
       - Object detection
       - Attention mechanism
       - Feature extraction

    2. Reasoning Layer (Symbolic):
       - Rule inference
       - Pattern matching
       - DSL operations

    3. Policy Layer (Hierarchical):
       - High-level: Operation selection
       - Mid-level: Object selection
       - Low-level: Execution

    4. Learning:
       - Supervised: Learn from human demonstrations
       - Reinforcement: Fine-tune with rewards
       - Meta-learning: Fast adaptation
    """

    def __init__(
        self,
        action_space_size: int = 9004,
        grid_size: int = 30,
        num_colors: int = 10,
        use_transformer: bool = True,
        use_hierarchical: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints/neuro_symbolic"
    ):
        """
        Args:
            action_space_size: Total number of actions
            grid_size: Maximum grid dimension
            num_colors: Number of ARC colors
            use_transformer: Use transformer attention
            use_hierarchical: Use hierarchical policy
            device: "cuda" or "cpu"
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__()

        self.action_space_size = action_space_size
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.device = torch.device(device)
        self.use_transformer = use_transformer
        self.use_hierarchical = use_hierarchical

        # ====================================
        # PERCEPTION LAYER
        # ====================================

        # Object detector (symbolic)
        self.object_detector = ObjectDetector(
            background_color=0,
            connectivity=4,
            min_object_size=1
        )

        # Transformer attention (neural)
        if use_transformer:
            self.transformer = TransformerAgent(
                grid_size=grid_size,
                num_colors=num_colors,
                d_model=128,
                num_layers=4,
                num_heads=8,
                num_actions=action_space_size
            ).to(self.device)
        else:
            self.transformer = None

        # ====================================
        # REASONING LAYER
        # ====================================

        # DSL interpreter
        self.dsl = ARCDSL()

        # Pattern matcher (placeholder)
        self.known_patterns = []

        # ====================================
        # POLICY LAYER
        # ====================================

        if use_hierarchical:
            self.hierarchical_policy = HierarchicalPolicy(
                grid_size=grid_size,
                num_colors=num_colors,
                device=str(self.device)
            )
        else:
            self.hierarchical_policy = None

        # ====================================
        # LEARNING
        # ====================================

        # Optimizers
        if use_transformer:
            self.transformer_optimizer = optim.Adam(
                self.transformer.parameters(),
                lr=1e-4
            )

        if use_hierarchical:
            # Separate optimizers for each level
            self.high_level_optimizer = optim.Adam(
                self.hierarchical_policy.high_level.parameters(),
                lr=1e-4
            )
            self.mid_level_optimizer = optim.Adam(
                self.hierarchical_policy.mid_level.parameters(),
                lr=1e-4
            )

        # Training stats
        self.training_steps = 0
        self.total_loss = 0.0

        # Mode
        self.mode = 'hybrid'  # 'transformer', 'hierarchical', or 'hybrid'

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[NEURO-SYMBOLIC AGENT] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Transformer: {use_transformer}")
        print(f"  Hierarchical: {use_hierarchical}")
        print(f"  Action space: {action_space_size}")

    def select_action(self, state: np.ndarray, info: Optional[Dict] = None) -> int:
        """
        Select action given current state

        Uses hybrid approach:
        1. Detect objects (perception)
        2. Extract features (perception)
        3. Hierarchical policy OR Transformer (depending on mode)

        Args:
            state: (H, W) grid
            info: Optional info dict

        Returns:
            action: Integer action ID
        """
        # 1. Perception: Detect objects
        objects = self.object_detector.detect_objects(state)
        grid_features = extract_grid_features(state)

        # 2. Policy selection based on mode
        if self.mode == 'hierarchical' and self.hierarchical_policy:
            # Use hierarchical policy
            output_grid, action_info = self.hierarchical_policy.act(
                state,
                mode='exploit'
            )

            # Convert output grid to action
            # (This is a simplification - in reality, we'd need to map grid changes to actions)
            action = self._grid_diff_to_action(state, output_grid)

        elif self.mode == 'transformer' and self.transformer:
            # Use transformer
            action = self.transformer.select_action(state, mode='exploit')

        elif self.mode == 'hybrid':
            # Use both and combine
            # Simple strategy: Use hierarchical if objects detected, else transformer

            if len(objects) > 0 and self.hierarchical_policy:
                output_grid, action_info = self.hierarchical_policy.act(state)
                action = self._grid_diff_to_action(state, output_grid)
            elif self.transformer:
                action = self.transformer.select_action(state)
            else:
                # Fallback: random action
                action = np.random.randint(0, self.action_space_size)

        else:
            # Fallback
            action = np.random.randint(0, self.action_space_size)

        return action

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ):
        """
        Update agent (RL update)

        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Episode done
            info: Optional info
        """
        # TODO: Implement RL update
        # For now, this is a placeholder
        pass

    def train_from_demonstrations(
        self,
        demo_buffer: DemonstrationBuffer,
        num_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train from human demonstrations

        Args:
            demo_buffer: Buffer with demonstrations
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        if not self.transformer:
            raise ValueError("Transformer must be enabled for training")

        self.transformer.train()

        epoch_losses = []

        for epoch in range(num_epochs):
            # Sample batch
            try:
                states, actions = demo_buffer.sample_batch(
                    batch_size=batch_size,
                    only_successful=True
                )
            except ValueError:
                print("[WARNING] No demonstration data available")
                break

            # Prepare tensors
            states_tensor = self._prepare_batch(states)
            actions_tensor = torch.LongTensor(actions).to(self.device)

            # Forward pass
            action_logits, _ = self.transformer(states_tensor)

            # Loss
            loss = nn.functional.cross_entropy(action_logits, actions_tensor)

            # Backward pass
            self.transformer_optimizer.zero_grad()
            loss.backward()
            self.transformer_optimizer.step()

            # Stats
            epoch_losses.append(loss.item())
            self.training_steps += 1
            self.total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        return {
            'avg_loss': avg_loss,
            'training_steps': self.training_steps
        }

    def _prepare_batch(self, states: np.ndarray) -> torch.Tensor:
        """Prepare batch of states for network"""
        if states.ndim == 2:
            states = states[np.newaxis, ...]

        batch_size = states.shape[0]
        h, w = states.shape[1], states.shape[2]

        # Pad to grid_size
        padded = np.zeros((batch_size, self.grid_size, self.grid_size), dtype=np.float32)
        padded[:, :h, :w] = states

        return torch.FloatTensor(padded).to(self.device)

    def _grid_diff_to_action(self, old_grid: np.ndarray, new_grid: np.ndarray) -> int:
        """
        Convert grid difference to action

        Finds the first changed cell and returns corresponding paint action
        """
        diff = (old_grid != new_grid)
        changed_positions = np.argwhere(diff)

        if len(changed_positions) == 0:
            # No change, return no-op (paint (0,0) with its current color)
            return 0

        # Take first changed position
        x, y = changed_positions[0]
        new_color = int(new_grid[x, y])

        # Convert to action: x * (30 * 10) + y * 10 + color
        action = x * (self.grid_size * self.num_colors) + y * self.num_colors + new_color

        return action

    def save(self, filepath: str):
        """Save agent checkpoint"""
        checkpoint = {
            'mode': self.mode,
            'training_steps': self.training_steps,
            'total_loss': self.total_loss
        }

        if self.transformer:
            checkpoint['transformer_state_dict'] = self.transformer.state_dict()
            checkpoint['transformer_optimizer_state_dict'] = self.transformer_optimizer.state_dict()

        if self.hierarchical_policy:
            checkpoint['high_level_state_dict'] = self.hierarchical_policy.high_level.state_dict()
            checkpoint['mid_level_state_dict'] = self.hierarchical_policy.mid_level.state_dict()
            checkpoint['high_level_optimizer_state_dict'] = self.high_level_optimizer.state_dict()
            checkpoint['mid_level_optimizer_state_dict'] = self.mid_level_optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"[NEURO-SYMBOLIC AGENT] Saved checkpoint to {filepath}")

    def load(self, filepath: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.mode = checkpoint.get('mode', 'hybrid')
        self.training_steps = checkpoint.get('training_steps', 0)
        self.total_loss = checkpoint.get('total_loss', 0.0)

        if self.transformer and 'transformer_state_dict' in checkpoint:
            self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])

        if self.hierarchical_policy:
            if 'high_level_state_dict' in checkpoint:
                self.hierarchical_policy.high_level.load_state_dict(checkpoint['high_level_state_dict'])
                self.high_level_optimizer.load_state_dict(checkpoint['high_level_optimizer_state_dict'])
            if 'mid_level_state_dict' in checkpoint:
                self.hierarchical_policy.mid_level.load_state_dict(checkpoint['mid_level_state_dict'])
                self.mid_level_optimizer.load_state_dict(checkpoint['mid_level_optimizer_state_dict'])

        print(f"[NEURO-SYMBOLIC AGENT] Loaded checkpoint from {filepath}")
        print(f"  Training steps: {self.training_steps}")

    def get_params(self) -> Dict:
        """Get agent parameters for visualization"""
        return {
            'agent_type': 'NeuroSymbolic',
            'mode': self.mode,
            'use_transformer': self.use_transformer,
            'use_hierarchical': self.use_hierarchical,
            'training_steps': self.training_steps,
            'avg_loss': self.total_loss / max(1, self.training_steps),
            'device': str(self.device)
        }

    def analyze_grid(self, grid: np.ndarray) -> Dict:
        """
        Comprehensive grid analysis

        Returns detailed information about the grid:
        - Objects detected
        - Grid features
        - Attention maps (if transformer enabled)
        - Suggested operations (if hierarchical enabled)
        """
        analysis = {}

        # Object detection
        objects = self.object_detector.detect_objects(grid)
        analysis['num_objects'] = len(objects)
        analysis['objects'] = [obj.to_dict() for obj in objects]

        # Grid features
        features = extract_grid_features(grid)
        analysis['features'] = features

        # Attention map
        if self.transformer:
            attention_map = self.transformer.visualize_attention(grid)
            analysis['attention_map'] = attention_map.tolist()

        # Suggested operation
        if self.hierarchical_policy:
            operation = self.hierarchical_policy.high_level.select_operation(grid)
            analysis['suggested_operation'] = operation.value

        return analysis
