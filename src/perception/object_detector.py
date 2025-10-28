"""
Object Detection Module for ARC Grids
Detects discrete objects in grids using connected components analysis
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from collections import defaultdict


@dataclass
class ARCObject:
    """Represents a detected object in an ARC grid"""
    object_id: int
    pixels: List[Tuple[int, int]]  # List of (row, col) positions
    color: int
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    size: int  # Number of pixels
    center: Tuple[float, float]  # Center of mass
    shape_type: str  # 'rectangle', 'line', 'blob', 'single'

    def get_width(self) -> int:
        """Get object width"""
        return self.bbox[3] - self.bbox[1] + 1

    def get_height(self) -> int:
        """Get object height"""
        return self.bbox[2] - self.bbox[0] + 1

    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)"""
        height = self.get_height()
        if height == 0:
            return 0.0
        return self.get_width() / height

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.object_id,
            'color': self.color,
            'size': self.size,
            'bbox': self.bbox,
            'center': self.center,
            'width': self.get_width(),
            'height': self.get_height(),
            'aspect_ratio': self.get_aspect_ratio(),
            'shape': self.shape_type
        }


class ObjectDetector:
    """
    Detects objects in ARC grids using connected components analysis

    Objects are defined as connected regions of the same color
    """

    def __init__(
        self,
        background_color: int = 0,
        connectivity: int = 4,  # 4 or 8 connectivity
        min_object_size: int = 1,
        merge_touching: bool = False
    ):
        """
        Args:
            background_color: Color to treat as background (default: 0 = black)
            connectivity: 4-connectivity or 8-connectivity
            min_object_size: Minimum pixels for valid object
            merge_touching: Whether to merge touching objects of same color
        """
        self.background_color = background_color
        self.connectivity = connectivity
        self.min_object_size = min_object_size
        self.merge_touching = merge_touching

    def detect_objects(self, grid: np.ndarray) -> List[ARCObject]:
        """
        Detect all objects in grid

        Args:
            grid: (H, W) numpy array with color values

        Returns:
            List of ARCObject instances
        """
        objects = []
        object_id = 0

        # Get unique colors (excluding background)
        unique_colors = np.unique(grid)
        unique_colors = unique_colors[unique_colors != self.background_color]

        # Detect objects for each color separately
        for color in unique_colors:
            color_objects = self._detect_objects_for_color(grid, color, object_id)
            objects.extend(color_objects)
            object_id += len(color_objects)

        return objects

    def _detect_objects_for_color(
        self,
        grid: np.ndarray,
        color: int,
        start_id: int
    ) -> List[ARCObject]:
        """Detect all objects of a specific color"""
        # Create binary mask for this color
        color_mask = (grid == color).astype(np.int32)

        # Connected components analysis
        if self.connectivity == 4:
            structure = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])
        else:  # 8-connectivity
            structure = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])

        labeled_array, num_features = ndimage.label(color_mask, structure=structure)

        # Extract each object
        objects = []
        for label_id in range(1, num_features + 1):
            obj = self._extract_object(
                labeled_array,
                label_id,
                color,
                start_id + label_id - 1
            )

            # Filter by size
            if obj.size >= self.min_object_size:
                objects.append(obj)

        return objects

    def _extract_object(
        self,
        labeled_array: np.ndarray,
        label_id: int,
        color: int,
        object_id: int
    ) -> ARCObject:
        """Extract object information from labeled array"""
        # Get pixels belonging to this object
        coords = np.argwhere(labeled_array == label_id)
        pixels = [(int(r), int(c)) for r, c in coords]

        # Bounding box
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        bbox = (int(min_row), int(min_col), int(max_row), int(max_col))

        # Size
        size = len(pixels)

        # Center of mass
        center_row = coords[:, 0].mean()
        center_col = coords[:, 1].mean()
        center = (float(center_row), float(center_col))

        # Shape classification
        shape_type = self._classify_shape(pixels, bbox)

        return ARCObject(
            object_id=object_id,
            pixels=pixels,
            color=color,
            bbox=bbox,
            size=size,
            center=center,
            shape_type=shape_type
        )

    def _classify_shape(
        self,
        pixels: List[Tuple[int, int]],
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """Classify object shape"""
        size = len(pixels)
        min_row, min_col, max_row, max_col = bbox
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        area = width * height

        # Single pixel
        if size == 1:
            return 'single'

        # Line (horizontal or vertical)
        if width == 1 and height > 1:
            return 'vertical_line'
        if height == 1 and width > 1:
            return 'horizontal_line'
        if width == 1 or height == 1:
            return 'line'

        # Rectangle (fills bounding box)
        fill_ratio = size / area
        if fill_ratio > 0.95:
            return 'rectangle'
        if fill_ratio > 0.8:
            return 'filled_shape'

        # Diagonal line
        if self._is_diagonal(pixels):
            return 'diagonal_line'

        # Default
        return 'blob'

    def _is_diagonal(self, pixels: List[Tuple[int, int]]) -> bool:
        """Check if pixels form a diagonal line"""
        if len(pixels) < 3:
            return False

        # Check if all pixels are on a diagonal
        pixels_sorted = sorted(pixels)
        first = pixels_sorted[0]

        # Check main diagonal (row+col constant) or anti-diagonal (row-col constant)
        main_diag = all((r - first[0]) == (c - first[1]) for r, c in pixels_sorted)
        anti_diag = all((r - first[0]) == -(c - first[1]) for r, c in pixels_sorted)

        return main_diag or anti_diag

    def detect_background_objects(self, grid: np.ndarray) -> List[ARCObject]:
        """
        Detect background regions (inverse detection)
        Useful for puzzles where background shapes are important
        """
        # Temporarily swap background color
        original_bg = self.background_color

        # Find most common color as new background
        unique, counts = np.unique(grid, return_counts=True)
        most_common = unique[counts.argmax()]

        if most_common == original_bg:
            # Background is already most common, detect everything else
            return self.detect_objects(grid)

        # Detect with swapped background
        self.background_color = most_common
        objects = self.detect_objects(grid)
        self.background_color = original_bg

        return objects

    def get_object_by_id(
        self,
        objects: List[ARCObject],
        object_id: int
    ) -> Optional[ARCObject]:
        """Get object by ID"""
        for obj in objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_objects_by_color(
        self,
        objects: List[ARCObject],
        color: int
    ) -> List[ARCObject]:
        """Get all objects of a specific color"""
        return [obj for obj in objects if obj.color == color]

    def get_objects_by_shape(
        self,
        objects: List[ARCObject],
        shape_type: str
    ) -> List[ARCObject]:
        """Get all objects of a specific shape"""
        return [obj for obj in objects if obj.shape_type == shape_type]

    def get_largest_object(self, objects: List[ARCObject]) -> Optional[ARCObject]:
        """Get largest object by size"""
        if not objects:
            return None
        return max(objects, key=lambda obj: obj.size)

    def get_smallest_object(self, objects: List[ARCObject]) -> Optional[ARCObject]:
        """Get smallest object by size"""
        if not objects:
            return None
        return min(objects, key=lambda obj: obj.size)

    def count_objects(self, objects: List[ARCObject]) -> int:
        """Count total objects"""
        return len(objects)

    def count_objects_by_color(self, objects: List[ARCObject]) -> Dict[int, int]:
        """Count objects grouped by color"""
        counts = defaultdict(int)
        for obj in objects:
            counts[obj.color] += 1
        return dict(counts)

    def get_spatial_relationships(
        self,
        obj1: ARCObject,
        obj2: ARCObject
    ) -> Dict[str, any]:
        """
        Compute spatial relationships between two objects

        Returns:
            Dictionary with relationship info:
            - 'relative_position': 'above', 'below', 'left', 'right', etc.
            - 'distance': Euclidean distance between centers
            - 'aligned': Whether objects are aligned horizontally/vertically
        """
        # Centers
        c1_row, c1_col = obj1.center
        c2_row, c2_col = obj2.center

        # Distance
        distance = np.sqrt((c1_row - c2_row)**2 + (c1_col - c2_col)**2)

        # Relative position
        delta_row = c2_row - c1_row
        delta_col = c2_col - c1_col

        if abs(delta_row) > abs(delta_col):
            # Vertical relationship
            rel_pos = 'below' if delta_row > 0 else 'above'
        else:
            # Horizontal relationship
            rel_pos = 'right' if delta_col > 0 else 'left'

        # Alignment
        aligned_horizontal = abs(delta_row) < 2  # Within 2 pixels
        aligned_vertical = abs(delta_col) < 2

        return {
            'relative_position': rel_pos,
            'distance': float(distance),
            'aligned_horizontal': aligned_horizontal,
            'aligned_vertical': aligned_vertical,
            'delta_row': float(delta_row),
            'delta_col': float(delta_col)
        }

    def visualize_objects(
        self,
        grid: np.ndarray,
        objects: List[ARCObject]
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes

        Returns:
            Grid with bounding boxes drawn (for debugging)
        """
        vis_grid = grid.copy()

        for obj in objects:
            # Draw bounding box (use color 9 = dark red for box)
            min_r, min_c, max_r, max_c = obj.bbox

            # Top and bottom edges
            for c in range(min_c, max_c + 1):
                if vis_grid[min_r, c] == self.background_color:
                    vis_grid[min_r, c] = 9
                if vis_grid[max_r, c] == self.background_color:
                    vis_grid[max_r, c] = 9

            # Left and right edges
            for r in range(min_r, max_r + 1):
                if vis_grid[r, min_c] == self.background_color:
                    vis_grid[r, min_c] = 9
                if vis_grid[r, max_c] == self.background_color:
                    vis_grid[r, max_c] = 9

        return vis_grid


# Utility functions

def detect_grid_symmetry(grid: np.ndarray) -> Dict[str, bool]:
    """
    Detect if grid has symmetry

    Returns:
        Dictionary with symmetry information:
        - 'horizontal': Grid is symmetric across horizontal axis
        - 'vertical': Grid is symmetric across vertical axis
        - 'rotational_90': Grid is symmetric under 90° rotation
        - 'rotational_180': Grid is symmetric under 180° rotation
    """
    h, w = grid.shape

    # Horizontal symmetry (flip up-down)
    horizontal = np.array_equal(grid, np.flipud(grid))

    # Vertical symmetry (flip left-right)
    vertical = np.array_equal(grid, np.fliplr(grid))

    # Rotational 90° (only for square grids)
    rotational_90 = False
    if h == w:
        rotational_90 = np.array_equal(grid, np.rot90(grid))

    # Rotational 180°
    rotational_180 = np.array_equal(grid, np.rot90(grid, 2))

    return {
        'horizontal': horizontal,
        'vertical': vertical,
        'rotational_90': rotational_90,
        'rotational_180': rotational_180
    }


def extract_grid_features(grid: np.ndarray) -> Dict:
    """
    Extract high-level features from grid

    Returns:
        Dictionary with grid features
    """
    h, w = grid.shape
    unique_colors = np.unique(grid)

    # Symmetry
    symmetry = detect_grid_symmetry(grid)

    # Color statistics
    color_counts = {int(c): int(np.sum(grid == c)) for c in unique_colors}

    # Sparsity (how empty is the grid)
    background_pixels = np.sum(grid == 0)
    sparsity = background_pixels / (h * w)

    return {
        'height': h,
        'width': w,
        'num_colors': len(unique_colors),
        'colors': [int(c) for c in unique_colors],
        'color_counts': color_counts,
        'sparsity': float(sparsity),
        'symmetry': symmetry,
        'is_square': h == w
    }
