"""
Hierarchical Policy for ARC Puzzles
Three-level decision making: What → Where → How
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from enum import Enum

from src.perception.object_detector import ARCObject, ObjectDetector
from src.symbolic.arc_dsl import ARCDSL, DSLObject, Direction, Axis


class HighLevelOperation(Enum):
    """High-level operations"""
    ROTATE_OBJECTS = 'rotate_objects'
    MIRROR_GRID = 'mirror_grid'
    MIRROR_OBJECTS = 'mirror_objects'
    COLOR_TRANSFORM = 'color_transform'
    FILL_PATTERN = 'fill_pattern'
    REPEAT_PATTERN = 'repeat_pattern'
    TRANSLATE_OBJECTS = 'translate_objects'
    SCALE_OBJECTS = 'scale_objects'
    COMPOSE_PATTERN = 'compose_pattern'
    NO_OP = 'no_op'


class ObjectSelector(Enum):
    """Mid-level object selection strategies"""
    ALL_OBJECTS = 'all_objects'
    BY_COLOR = 'by_color'
    BY_SIZE = 'by_size'
    LARGEST = 'largest'
    SMALLEST = 'smallest'
    BY_POSITION = 'by_position'
    NONE = 'none'


class HighLevelPolicy(nn.Module):
    """
    High-Level Policy Network
    Decides WHAT operation to perform
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        hidden_dim: int = 256,
        num_operations: int = 10
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_operations = num_operations

        # CNN feature extractor
        self.conv1 = nn.Conv2d(num_colors, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Operation classifier
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_operations)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            grid: (batch, grid_size, grid_size) with values 0-9

        Returns:
            operation_logits: (batch, num_operations)
        """
        # One-hot encode
        grid_onehot = F.one_hot(grid.long(), num_classes=self.num_colors)
        grid_onehot = grid_onehot.permute(0, 3, 1, 2).float()  # (batch, 10, H, W)

        # CNN
        x = F.relu(self.bn1(self.conv1(grid_onehot)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Pool
        x = self.global_pool(x).view(x.size(0), -1)

        # Classify
        operation_logits = self.fc(x)

        return operation_logits

    def select_operation(
        self,
        grid: np.ndarray,
        mode: str = 'exploit'
    ) -> HighLevelOperation:
        """
        Select high-level operation

        Args:
            grid: (H, W) numpy array
            mode: 'exploit' or 'explore'

        Returns:
            Selected operation
        """
        # Pad to grid_size
        h, w = grid.shape
        padded = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        padded[:h, :w] = grid

        # To tensor
        grid_tensor = torch.FloatTensor(padded).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(grid_tensor)[0]

            if mode == 'explore':
                # Sample from distribution
                probs = F.softmax(logits, dim=0)
                action_idx = torch.multinomial(probs, 1).item()
            else:
                # Greedy
                action_idx = torch.argmax(logits).item()

        # Map to operation
        operations = list(HighLevelOperation)
        return operations[action_idx]


class MidLevelPolicy(nn.Module):
    """
    Mid-Level Policy Network
    Decides WHERE to apply operation (which objects)
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        hidden_dim: int = 128,
        num_selectors: int = 7
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_selectors = num_selectors

        # Simpler CNN
        self.conv1 = nn.Conv2d(num_colors, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Selector classifier
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_selectors)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            grid: (batch, grid_size, grid_size)

        Returns:
            selector_logits: (batch, num_selectors)
        """
        # One-hot encode
        grid_onehot = F.one_hot(grid.long(), num_classes=self.num_colors)
        grid_onehot = grid_onehot.permute(0, 3, 1, 2).float()

        # CNN
        x = F.relu(self.conv1(grid_onehot))
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)

        # Classify
        selector_logits = self.fc(x)

        return selector_logits

    def select_objects(
        self,
        grid: np.ndarray,
        objects: List[ARCObject],
        mode: str = 'exploit'
    ) -> Tuple[ObjectSelector, List[ARCObject]]:
        """
        Select which objects to operate on

        Args:
            grid: (H, W) numpy array
            objects: List of detected objects
            mode: 'exploit' or 'explore'

        Returns:
            (selector_type, selected_objects)
        """
        if not objects:
            return ObjectSelector.NONE, []

        # Pad grid
        h, w = grid.shape
        padded = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        padded[:h, :w] = grid

        # To tensor
        grid_tensor = torch.FloatTensor(padded).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(grid_tensor)[0]

            if mode == 'explore':
                probs = F.softmax(logits, dim=0)
                selector_idx = torch.multinomial(probs, 1).item()
            else:
                selector_idx = torch.argmax(logits).item()

        # Map to selector
        selectors = list(ObjectSelector)
        selector_type = selectors[selector_idx]

        # Apply selection
        selected = self._apply_selector(selector_type, objects)

        return selector_type, selected

    def _apply_selector(
        self,
        selector: ObjectSelector,
        objects: List[ARCObject]
    ) -> List[ARCObject]:
        """Apply selection strategy"""
        if selector == ObjectSelector.ALL_OBJECTS:
            return objects

        elif selector == ObjectSelector.BY_COLOR:
            # Select most common color
            colors = [obj.color for obj in objects]
            if colors:
                most_common = max(set(colors), key=colors.count)
                return [obj for obj in objects if obj.color == most_common]
            return []

        elif selector == ObjectSelector.BY_SIZE:
            # Select medium-sized objects
            if not objects:
                return []
            sizes = [obj.size for obj in objects]
            median_size = np.median(sizes)
            return [obj for obj in objects if abs(obj.size - median_size) < median_size * 0.3]

        elif selector == ObjectSelector.LARGEST:
            if objects:
                largest = max(objects, key=lambda obj: obj.size)
                return [largest]
            return []

        elif selector == ObjectSelector.SMALLEST:
            if objects:
                smallest = min(objects, key=lambda obj: obj.size)
                return [smallest]
            return []

        elif selector == ObjectSelector.BY_POSITION:
            # Select objects in top-left quadrant
            if not objects:
                return []
            centers = [obj.center for obj in objects]
            median_r = np.median([r for r, c in centers])
            median_c = np.median([c for r, c in centers])
            return [obj for obj in objects
                   if obj.center[0] < median_r and obj.center[1] < median_c]

        elif selector == ObjectSelector.NONE:
            return []

        return []


class LowLevelExecutor:
    """
    Low-Level Executor
    Executes DSL operations on grid
    """

    def __init__(self):
        self.dsl = ARCDSL()
        self.detector = ObjectDetector()

    def execute(
        self,
        grid: np.ndarray,
        operation: HighLevelOperation,
        objects: List[ARCObject],
        parameters: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Execute operation on grid

        Args:
            grid: Input grid
            operation: High-level operation
            objects: Selected objects
            parameters: Operation parameters

        Returns:
            Output grid
        """
        if parameters is None:
            parameters = {}

        # Convert ARCObject to DSLObject
        dsl_objects = [
            DSLObject(
                pixels=obj.pixels,
                color=obj.color,
                bbox=obj.bbox
            )
            for obj in objects
        ]

        # Execute based on operation
        if operation == HighLevelOperation.ROTATE_OBJECTS:
            return self._execute_rotate_objects(grid, dsl_objects, parameters)

        elif operation == HighLevelOperation.MIRROR_GRID:
            return self._execute_mirror_grid(grid, parameters)

        elif operation == HighLevelOperation.MIRROR_OBJECTS:
            return self._execute_mirror_objects(grid, dsl_objects, parameters)

        elif operation == HighLevelOperation.COLOR_TRANSFORM:
            return self._execute_color_transform(grid, parameters)

        elif operation == HighLevelOperation.FILL_PATTERN:
            return self._execute_fill_pattern(grid, parameters)

        elif operation == HighLevelOperation.REPEAT_PATTERN:
            return self._execute_repeat_pattern(grid, parameters)

        elif operation == HighLevelOperation.TRANSLATE_OBJECTS:
            return self._execute_translate_objects(grid, dsl_objects, parameters)

        elif operation == HighLevelOperation.SCALE_OBJECTS:
            return self._execute_scale_objects(grid, dsl_objects, parameters)

        elif operation == HighLevelOperation.NO_OP:
            return grid

        else:
            # Default: return unchanged
            return grid

    def _execute_rotate_objects(
        self,
        grid: np.ndarray,
        objects: List[DSLObject],
        params: Dict
    ) -> np.ndarray:
        """Rotate objects"""
        angle = params.get('angle', 90)

        rotated_objects = [
            self.dsl.rotate_object(obj, angle)
            for obj in objects
        ]

        return self.dsl.compose_objects(rotated_objects, grid.shape)

    def _execute_mirror_grid(self, grid: np.ndarray, params: Dict) -> np.ndarray:
        """Mirror entire grid"""
        axis = params.get('axis', Axis.HORIZONTAL)
        return self.dsl.mirror_grid(grid, axis)

    def _execute_mirror_objects(
        self,
        grid: np.ndarray,
        objects: List[DSLObject],
        params: Dict
    ) -> np.ndarray:
        """Mirror objects"""
        axis = params.get('axis', Axis.HORIZONTAL)

        mirrored_objects = [
            self.dsl.mirror_object(obj, axis)
            for obj in objects
        ]

        return self.dsl.compose_objects(mirrored_objects, grid.shape)

    def _execute_color_transform(self, grid: np.ndarray, params: Dict) -> np.ndarray:
        """Transform colors"""
        color_map = params.get('color_map', {1: 2, 2: 1})  # Default: swap blue and red
        return self.dsl.apply_color_map(grid, color_map)

    def _execute_fill_pattern(self, grid: np.ndarray, params: Dict) -> np.ndarray:
        """Fill with pattern"""
        color = params.get('color', 1)
        return self.dsl.fill_region(grid, [], color)

    def _execute_repeat_pattern(self, grid: np.ndarray, params: Dict) -> np.ndarray:
        """Repeat pattern"""
        times = params.get('times', 2)
        direction = params.get('direction', Direction.RIGHT)
        return self.dsl.repeat_pattern(grid, times, direction)

    def _execute_translate_objects(
        self,
        grid: np.ndarray,
        objects: List[DSLObject],
        params: Dict
    ) -> np.ndarray:
        """Translate objects"""
        delta_row = params.get('delta_row', 1)
        delta_col = params.get('delta_col', 1)

        translated_objects = [
            self.dsl.translate_object(obj, delta_row, delta_col)
            for obj in objects
        ]

        return self.dsl.compose_objects(translated_objects, grid.shape)

    def _execute_scale_objects(
        self,
        grid: np.ndarray,
        objects: List[DSLObject],
        params: Dict
    ) -> np.ndarray:
        """Scale objects"""
        scale = params.get('scale', 2)

        scaled_objects = [
            self.dsl.scale_object(obj, scale)
            for obj in objects
        ]

        return self.dsl.compose_objects(scaled_objects, grid.shape)


class HierarchicalPolicy:
    """
    Complete Hierarchical Policy System
    Combines all three levels
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        device: str = 'cpu'
    ):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.device = device

        # Three levels
        self.high_level = HighLevelPolicy(grid_size, num_colors).to(device)
        self.mid_level = MidLevelPolicy(grid_size, num_colors).to(device)
        self.low_level = LowLevelExecutor()

        # Object detector
        self.detector = ObjectDetector()

    def act(
        self,
        grid: np.ndarray,
        mode: str = 'exploit',
        parameters: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete hierarchical action

        Args:
            grid: Input grid
            mode: 'exploit' or 'explore'
            parameters: Optional parameters for low-level execution

        Returns:
            (output_grid, action_info)
        """
        # 1. Detect objects
        objects = self.detector.detect_objects(grid)

        # 2. High-level: What to do?
        operation = self.high_level.select_operation(grid, mode)

        # 3. Mid-level: Where to do it?
        selector, selected_objects = self.mid_level.select_objects(grid, objects, mode)

        # 4. Low-level: How to do it?
        output_grid = self.low_level.execute(grid, operation, selected_objects, parameters)

        # Action info
        action_info = {
            'operation': operation.value,
            'selector': selector.value,
            'num_objects_detected': len(objects),
            'num_objects_selected': len(selected_objects),
            'parameters': parameters or {}
        }

        return output_grid, action_info

    def train_high_level(self, train_data, optimizer, epochs=10):
        """Train high-level policy (supervised)"""
        self.high_level.train()
        # TODO: Implement training loop
        pass

    def train_mid_level(self, train_data, optimizer, epochs=10):
        """Train mid-level policy (supervised)"""
        self.mid_level.train()
        # TODO: Implement training loop
        pass
