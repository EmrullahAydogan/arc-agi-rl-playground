"""
ARC Domain Specific Language (DSL)
High-level operations for ARC puzzle transformations
"""
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy


class Direction(Enum):
    """Cardinal directions"""
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


class Axis(Enum):
    """Symmetry axes"""
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    BOTH = 'both'


@dataclass
class DSLObject:
    """Object representation for DSL operations"""
    pixels: List[Tuple[int, int]]  # List of (row, col)
    color: int
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

    def to_mask(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Convert object to binary mask"""
        mask = np.zeros(grid_shape, dtype=bool)
        for r, c in self.pixels:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                mask[r, c] = True
        return mask


class ARCDSL:
    """
    ARC Domain Specific Language
    High-level operations for grid transformations
    """

    def __init__(self):
        """Initialize DSL"""
        self.operation_history = []

    # ========================================
    # GRID OPERATIONS
    # ========================================

    def create_empty_grid(self, height: int, width: int, color: int = 0) -> np.ndarray:
        """Create empty grid filled with color"""
        return np.full((height, width), color, dtype=np.int32)

    def copy_grid(self, grid: np.ndarray) -> np.ndarray:
        """Deep copy of grid"""
        return grid.copy()

    def get_grid_size(self, grid: np.ndarray) -> Tuple[int, int]:
        """Get (height, width) of grid"""
        return grid.shape

    def resize_grid(
        self,
        grid: np.ndarray,
        new_height: int,
        new_width: int,
        fill_color: int = 0
    ) -> np.ndarray:
        """Resize grid, padding or cropping as needed"""
        h, w = grid.shape
        new_grid = np.full((new_height, new_width), fill_color, dtype=np.int32)

        # Copy overlapping region
        copy_h = min(h, new_height)
        copy_w = min(w, new_width)
        new_grid[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        return new_grid

    def crop_grid(
        self,
        grid: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop grid to bounding box"""
        min_r, min_c, max_r, max_c = bbox
        return grid[min_r:max_r+1, min_c:max_c+1].copy()

    def extend_grid(
        self,
        grid: np.ndarray,
        direction: Direction,
        amount: int,
        fill_color: int = 0
    ) -> np.ndarray:
        """Extend grid in direction"""
        h, w = grid.shape

        if direction == Direction.UP:
            new_rows = np.full((amount, w), fill_color, dtype=np.int32)
            return np.vstack([new_rows, grid])
        elif direction == Direction.DOWN:
            new_rows = np.full((amount, w), fill_color, dtype=np.int32)
            return np.vstack([grid, new_rows])
        elif direction == Direction.LEFT:
            new_cols = np.full((h, amount), fill_color, dtype=np.int32)
            return np.hstack([new_cols, grid])
        elif direction == Direction.RIGHT:
            new_cols = np.full((h, amount), fill_color, dtype=np.int32)
            return np.hstack([grid, new_cols])

    # ========================================
    # OBJECT OPERATIONS
    # ========================================

    def rotate_object(
        self,
        obj: DSLObject,
        angle: int,
        center: Optional[Tuple[float, float]] = None
    ) -> DSLObject:
        """
        Rotate object by angle (90, 180, 270 degrees)

        Args:
            obj: Object to rotate
            angle: Rotation angle (90, 180, 270, -90)
            center: Rotation center (default: object center)
        """
        if angle not in [90, 180, 270, -90]:
            raise ValueError("Angle must be 90, 180, 270, or -90")

        # Normalize angle
        angle = angle % 360

        # Get object center if not provided
        if center is None:
            min_r, min_c, max_r, max_c = obj.bbox
            center = ((min_r + max_r) / 2, (min_c + max_c) / 2)

        center_r, center_c = center

        # Rotate pixels
        new_pixels = []
        for r, c in obj.pixels:
            # Translate to origin
            r_rel = r - center_r
            c_rel = c - center_c

            # Rotate
            if angle == 90:
                r_new = -c_rel
                c_new = r_rel
            elif angle == 180:
                r_new = -r_rel
                c_new = -c_rel
            elif angle == 270:
                r_new = c_rel
                c_new = -r_rel

            # Translate back
            r_final = int(round(r_new + center_r))
            c_final = int(round(c_new + center_c))

            new_pixels.append((r_final, c_final))

        # Compute new bounding box
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = (min(rows), min(cols), max(rows), max(cols))

        return DSLObject(pixels=new_pixels, color=obj.color, bbox=new_bbox)

    def mirror_object(self, obj: DSLObject, axis: Axis) -> DSLObject:
        """Mirror object across axis"""
        min_r, min_c, max_r, max_c = obj.bbox

        if axis == Axis.HORIZONTAL:
            # Mirror horizontally (flip left-right)
            center_c = (min_c + max_c) / 2
            new_pixels = [(r, int(2 * center_c - c)) for r, c in obj.pixels]
        elif axis == Axis.VERTICAL:
            # Mirror vertically (flip up-down)
            center_r = (min_r + max_r) / 2
            new_pixels = [(int(2 * center_r - r), c) for r, c in obj.pixels]
        elif axis == Axis.BOTH:
            # Mirror both axes
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2
            new_pixels = [
                (int(2 * center_r - r), int(2 * center_c - c))
                for r, c in obj.pixels
            ]

        # Compute new bounding box
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = (min(rows), min(cols), max(rows), max(cols))

        return DSLObject(pixels=new_pixels, color=obj.color, bbox=new_bbox)

    def translate_object(
        self,
        obj: DSLObject,
        delta_row: int,
        delta_col: int
    ) -> DSLObject:
        """Translate object by delta"""
        new_pixels = [(r + delta_row, c + delta_col) for r, c in obj.pixels]

        # Update bbox
        min_r, min_c, max_r, max_c = obj.bbox
        new_bbox = (
            min_r + delta_row,
            min_c + delta_col,
            max_r + delta_row,
            max_c + delta_col
        )

        return DSLObject(pixels=new_pixels, color=obj.color, bbox=new_bbox)

    def scale_object(
        self,
        obj: DSLObject,
        scale_factor: int
    ) -> DSLObject:
        """Scale object by integer factor"""
        if scale_factor < 1:
            raise ValueError("Scale factor must be >= 1")

        # Scale pixels
        new_pixels = []
        for r, c in obj.pixels:
            for dr in range(scale_factor):
                for dc in range(scale_factor):
                    new_pixels.append((r * scale_factor + dr, c * scale_factor + dc))

        # Compute new bbox
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = (min(rows), min(cols), max(rows), max(cols))

        return DSLObject(pixels=new_pixels, color=obj.color, bbox=new_bbox)

    # ========================================
    # COLOR OPERATIONS
    # ========================================

    def recolor_object(self, obj: DSLObject, new_color: int) -> DSLObject:
        """Change object color"""
        return DSLObject(pixels=obj.pixels, color=new_color, bbox=obj.bbox)

    def apply_color_map(
        self,
        grid: np.ndarray,
        color_mapping: Dict[int, int]
    ) -> np.ndarray:
        """Apply color mapping to grid"""
        result = grid.copy()
        for old_color, new_color in color_mapping.items():
            result[grid == old_color] = new_color
        return result

    def swap_colors(
        self,
        grid: np.ndarray,
        color1: int,
        color2: int
    ) -> np.ndarray:
        """Swap two colors in grid"""
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result

    # ========================================
    # FILL OPERATIONS
    # ========================================

    def fill_region(
        self,
        grid: np.ndarray,
        region: List[Tuple[int, int]],
        color: int
    ) -> np.ndarray:
        """Fill region with color"""
        result = grid.copy()
        for r, c in region:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                result[r, c] = color
        return result

    def flood_fill(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        new_color: int
    ) -> np.ndarray:
        """Flood fill from starting position"""
        from scipy.ndimage import label

        result = grid.copy()
        r, c = start_pos

        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return result

        old_color = grid[r, c]
        if old_color == new_color:
            return result

        # Find connected region with same color
        mask = (grid == old_color)
        labeled, _ = label(mask)

        region_label = labeled[r, c]
        if region_label == 0:
            return result

        # Fill the region
        result[labeled == region_label] = new_color

        return result

    # ========================================
    # PATTERN OPERATIONS
    # ========================================

    def repeat_pattern(
        self,
        pattern: np.ndarray,
        times: int,
        direction: Direction
    ) -> np.ndarray:
        """Repeat pattern in direction"""
        if direction in [Direction.UP, Direction.DOWN]:
            # Vertical repetition
            result = np.tile(pattern, (times, 1))
        else:  # LEFT or RIGHT
            # Horizontal repetition
            result = np.tile(pattern, (1, times))

        return result

    def tile_pattern(
        self,
        pattern: np.ndarray,
        target_height: int,
        target_width: int
    ) -> np.ndarray:
        """Tile pattern to fill target size"""
        p_h, p_w = pattern.shape

        # Calculate repetitions needed
        times_h = (target_height + p_h - 1) // p_h
        times_w = (target_width + p_w - 1) // p_w

        # Tile
        tiled = np.tile(pattern, (times_h, times_w))

        # Crop to exact size
        return tiled[:target_height, :target_width]

    # ========================================
    # COMPOSITION OPERATIONS
    # ========================================

    def compose_objects(
        self,
        objects: List[DSLObject],
        grid_shape: Tuple[int, int],
        background_color: int = 0
    ) -> np.ndarray:
        """Compose multiple objects onto grid"""
        grid = np.full(grid_shape, background_color, dtype=np.int32)

        for obj in objects:
            for r, c in obj.pixels:
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    grid[r, c] = obj.color

        return grid

    def overlay_grids(
        self,
        base_grid: np.ndarray,
        overlay_grid: np.ndarray,
        position: Tuple[int, int] = (0, 0),
        transparent_color: Optional[int] = 0
    ) -> np.ndarray:
        """Overlay one grid on another"""
        result = base_grid.copy()
        r_offset, c_offset = position
        o_h, o_w = overlay_grid.shape

        for r in range(o_h):
            for c in range(o_w):
                target_r = r + r_offset
                target_c = c + c_offset

                if (0 <= target_r < result.shape[0] and
                    0 <= target_c < result.shape[1]):

                    # Skip transparent color
                    if transparent_color is not None:
                        if overlay_grid[r, c] != transparent_color:
                            result[target_r, target_c] = overlay_grid[r, c]
                    else:
                        result[target_r, target_c] = overlay_grid[r, c]

        return result

    # ========================================
    # GRID TRANSFORMATIONS
    # ========================================

    def rotate_grid(self, grid: np.ndarray, angle: int) -> np.ndarray:
        """Rotate entire grid"""
        if angle == 90:
            return np.rot90(grid, k=-1)  # Clockwise
        elif angle == 180:
            return np.rot90(grid, k=2)
        elif angle == 270 or angle == -90:
            return np.rot90(grid, k=1)  # Counter-clockwise
        else:
            raise ValueError("Angle must be 90, 180, or 270")

    def mirror_grid(self, grid: np.ndarray, axis: Axis) -> np.ndarray:
        """Mirror entire grid"""
        if axis == Axis.HORIZONTAL:
            return np.fliplr(grid)
        elif axis == Axis.VERTICAL:
            return np.flipud(grid)
        elif axis == Axis.BOTH:
            return np.flipud(np.fliplr(grid))

    def transpose_grid(self, grid: np.ndarray) -> np.ndarray:
        """Transpose grid (swap rows and columns)"""
        return grid.T

    # ========================================
    # CONDITIONAL OPERATIONS
    # ========================================

    def filter_objects(
        self,
        objects: List[DSLObject],
        predicate: Callable[[DSLObject], bool]
    ) -> List[DSLObject]:
        """Filter objects by predicate"""
        return [obj for obj in objects if predicate(obj)]

    def map_objects(
        self,
        objects: List[DSLObject],
        transform: Callable[[DSLObject], DSLObject]
    ) -> List[DSLObject]:
        """Apply transformation to each object"""
        return [transform(obj) for obj in objects]

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    def get_object_count(self, objects: List[DSLObject]) -> int:
        """Count objects"""
        return len(objects)

    def get_largest_object(self, objects: List[DSLObject]) -> Optional[DSLObject]:
        """Get largest object by pixel count"""
        if not objects:
            return None
        return max(objects, key=lambda obj: len(obj.pixels))

    def get_smallest_object(self, objects: List[DSLObject]) -> Optional[DSLObject]:
        """Get smallest object by pixel count"""
        if not objects:
            return None
        return min(objects, key=lambda obj: len(obj.pixels))

    def get_objects_by_color(
        self,
        objects: List[DSLObject],
        color: int
    ) -> List[DSLObject]:
        """Get objects of specific color"""
        return [obj for obj in objects if obj.color == color]

    def objects_to_grid(
        self,
        objects: List[DSLObject],
        grid_shape: Tuple[int, int],
        background_color: int = 0
    ) -> np.ndarray:
        """Alias for compose_objects"""
        return self.compose_objects(objects, grid_shape, background_color)


# Global DSL instance
dsl = ARCDSL()


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def execute_program(program: str, input_grid: np.ndarray, dsl_instance: ARCDSL = None) -> np.ndarray:
    """
    Execute DSL program on input grid

    Args:
        program: DSL program string
        input_grid: Input grid
        dsl_instance: DSL instance (uses global if None)

    Returns:
        Output grid
    """
    if dsl_instance is None:
        dsl_instance = dsl

    # TODO: Implement program parser and executor
    # For now, this is a placeholder
    raise NotImplementedError("Program execution not yet implemented")
