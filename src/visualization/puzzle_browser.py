#!/usr/bin/env python3
"""
Puzzle Browser - Browse and select ARC puzzles with thumbnail previews
"""
import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np


class PuzzleThumbnail:
    """Thumbnail widget for a single puzzle"""

    # ARC color palette (same as pygame viewer)
    COLORS = [
        (0, 0, 0),         # 0: Black
        (0, 116, 217),     # 1: Blue
        (255, 65, 54),     # 2: Red
        (46, 204, 64),     # 3: Green
        (255, 220, 0),     # 4: Yellow
        (170, 170, 170),   # 5: Gray
        (240, 18, 190),    # 6: Magenta
        (255, 133, 27),    # 7: Orange
        (127, 219, 255),   # 8: Light Blue
        (135, 12, 37),     # 9: Dark Red
    ]

    def __init__(
        self,
        parent: Canvas,
        puzzle_id: str,
        puzzle_data: dict,
        x: int,
        y: int,
        width: int = 150,
        height: int = 180,
        on_click: Optional[Callable] = None
    ):
        """
        Args:
            parent: Parent canvas
            puzzle_id: Puzzle identifier
            puzzle_data: Puzzle data dict
            x, y: Position in canvas
            width, height: Thumbnail dimensions
            on_click: Callback when clicked (receives puzzle_id)
        """
        self.parent = parent
        self.puzzle_id = puzzle_id
        self.puzzle_data = puzzle_data
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.on_click = on_click

        # Extract info
        self.num_train = len(puzzle_data.get('train', []))
        self.num_test = len(puzzle_data.get('test', []))

        # Get first train input grid for thumbnail
        if self.num_train > 0:
            self.grid = np.array(puzzle_data['train'][0]['input'])
        else:
            self.grid = np.array([[0]])  # Empty grid fallback

        self.grid_height, self.grid_width = self.grid.shape

        # Create visual elements
        self._create_widgets()

    def _create_widgets(self):
        """Create thumbnail widgets on canvas"""
        # Tag prefix to avoid confusion with item IDs
        self.tag = f"puzzle_{self.puzzle_id}"

        # Border/background rectangle
        self.border_rect = self.parent.create_rectangle(
            self.x, self.y,
            self.x + self.width, self.y + self.height,
            fill='#2b2b2b',
            outline='#555555',
            width=2,
            tags=('thumbnail', self.tag)
        )

        # Grid preview area (top 120px)
        grid_area_height = 120
        self._render_grid_thumbnail(
            self.x + 5,
            self.y + 5,
            self.width - 10,
            grid_area_height - 10
        )

        # Info area (bottom 60px)
        info_y = self.y + grid_area_height

        # Puzzle ID text
        self.parent.create_text(
            self.x + self.width // 2,
            info_y + 10,
            text=self.puzzle_id[:8] + "..." if len(self.puzzle_id) > 8 else self.puzzle_id,
            fill='white',
            font=('monospace', 10, 'bold'),
            tags=('thumbnail', self.tag)
        )

        # Grid size info
        self.parent.create_text(
            self.x + self.width // 2,
            info_y + 28,
            text=f"Grid: {self.grid_width}x{self.grid_height}",
            fill='#aaaaaa',
            font=('monospace', 8),
            tags=('thumbnail', self.tag)
        )

        # Sample count info
        self.parent.create_text(
            self.x + self.width // 2,
            info_y + 44,
            text=f"Train: {self.num_train} | Test: {self.num_test}",
            fill='#aaaaaa',
            font=('monospace', 8),
            tags=('thumbnail', self.tag)
        )

        # Bind events to this thumbnail's tag (after all items are created)
        self.parent.tag_bind(self.tag, '<Button-1>', self._on_click)
        self.parent.tag_bind(self.tag, '<Enter>', self._on_hover_enter)
        self.parent.tag_bind(self.tag, '<Leave>', self._on_hover_leave)

    def _render_grid_thumbnail(self, x: int, y: int, width: int, height: int):
        """Render the grid as a small thumbnail"""
        # Calculate cell size to fit in the area
        cell_width = width / self.grid_width
        cell_height = height / self.grid_height
        cell_size = min(cell_width, cell_height, 10)  # Max 10px per cell

        # Center the grid in the area
        grid_pixel_width = self.grid_width * cell_size
        grid_pixel_height = self.grid_height * cell_size
        start_x = x + (width - grid_pixel_width) / 2
        start_y = y + (height - grid_pixel_height) / 2

        # Draw each cell
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                color_idx = int(self.grid[row, col])
                rgb = self.COLORS[color_idx]
                color_hex = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

                cell_x = start_x + col * cell_size
                cell_y = start_y + row * cell_size

                self.parent.create_rectangle(
                    cell_x, cell_y,
                    cell_x + cell_size, cell_y + cell_size,
                    fill=color_hex,
                    outline='#444444' if cell_size >= 4 else '',
                    width=1 if cell_size >= 4 else 0,
                    tags=('thumbnail', self.tag, 'grid')
                )

    def _on_click(self, event):
        """Handle click event"""
        if self.on_click:
            self.on_click(self.puzzle_id)

    def _on_hover_enter(self, event):
        """Handle hover enter"""
        self.parent.itemconfig(self.border_rect, outline='#00ff00', width=3)

    def _on_hover_leave(self, event):
        """Handle hover leave"""
        self.parent.itemconfig(self.border_rect, outline='#555555', width=2)

    def highlight(self, selected: bool):
        """Highlight as selected"""
        if selected:
            self.parent.itemconfig(self.border_rect, outline='#ffff00', width=4)
        else:
            self.parent.itemconfig(self.border_rect, outline='#555555', width=2)


class PuzzleBrowser:
    """Browse and select ARC puzzles"""

    def __init__(
        self,
        data_loader,
        on_puzzle_select: Optional[Callable] = None,
        title: str = "ARC Puzzle Browser"
    ):
        """
        Args:
            data_loader: ARCDataLoader instance
            on_puzzle_select: Callback when puzzle is selected (receives puzzle_id, dataset)
            title: Window title
        """
        self.data_loader = data_loader
        self.on_puzzle_select = on_puzzle_select
        self.title = title

        self.current_dataset = 'training'
        self.selected_puzzle_id = None
        self.thumbnails: Dict[str, PuzzleThumbnail] = {}

        # Create window
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("820x600")
        self.window.configure(bg='#1e1e1e')

        # Create UI
        self._create_ui()

        # Load initial puzzles
        self._load_puzzles()

        print(f"[PUZZLE BROWSER] Window created: {title}")

    def _create_ui(self):
        """Create the UI layout"""
        # Top toolbar
        toolbar = tk.Frame(self.window, bg='#2b2b2b', height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Dataset selector
        tk.Label(
            toolbar,
            text="Dataset:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)

        self.dataset_var = tk.StringVar(value='training')
        dataset_combo = ttk.Combobox(
            toolbar,
            textvariable=self.dataset_var,
            values=['training', 'evaluation'],
            state='readonly',
            width=15
        )
        dataset_combo.pack(side=tk.LEFT, padx=5)
        dataset_combo.bind('<<ComboboxSelected>>', self._on_dataset_change)

        # Search box
        tk.Label(
            toolbar,
            text="Search:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(
            toolbar,
            textvariable=self.search_var,
            width=20,
            bg='#3c3c3c',
            fg='white',
            insertbackground='white'
        )
        search_entry.pack(side=tk.LEFT, padx=5)
        self.search_var.trace('w', self._on_search_change)

        # Puzzle count label
        self.count_label = tk.Label(
            toolbar,
            text="Puzzles: 0",
            bg='#2b2b2b',
            fg='#aaaaaa',
            font=('Arial', 10)
        )
        self.count_label.pack(side=tk.RIGHT, padx=10)

        # Canvas with scrollbar for thumbnails
        canvas_frame = tk.Frame(self.window, bg='#1e1e1e')
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = Canvas(
            canvas_frame,
            bg='#1e1e1e',
            highlightthickness=0
        )

        scrollbar = Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)  # Linux scroll up
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)  # Linux scroll down

    def _load_puzzles(self):
        """Load puzzles from current dataset"""
        # Clear existing thumbnails
        self.canvas.delete('all')
        self.thumbnails.clear()

        # Get puzzle list
        puzzle_ids = self.data_loader.get_all_puzzle_ids(self.current_dataset)

        # Filter by search term
        search_term = self.search_var.get().lower()
        if search_term:
            puzzle_ids = [pid for pid in puzzle_ids if search_term in pid.lower()]

        # Update count
        self.count_label.config(text=f"Puzzles: {len(puzzle_ids)}")

        # Layout parameters
        cols = 5
        thumb_width = 150
        thumb_height = 180
        spacing = 10

        # Create thumbnails in grid
        for idx, puzzle_id in enumerate(puzzle_ids):
            row = idx // cols
            col = idx % cols

            x = col * (thumb_width + spacing) + spacing
            y = row * (thumb_height + spacing) + spacing

            # Load puzzle data
            puzzle_data = self.data_loader.get_puzzle(puzzle_id, self.current_dataset)
            if puzzle_data is None:
                continue

            # Create thumbnail
            thumbnail = PuzzleThumbnail(
                self.canvas,
                puzzle_id,
                puzzle_data,
                x, y,
                thumb_width,
                thumb_height,
                on_click=self._on_puzzle_click
            )
            self.thumbnails[puzzle_id] = thumbnail

        # Update canvas scroll region
        max_rows = (len(puzzle_ids) + cols - 1) // cols
        scroll_height = max_rows * (thumb_height + spacing) + spacing
        self.canvas.configure(scrollregion=(0, 0, 800, scroll_height))

    def _on_dataset_change(self, event):
        """Handle dataset selection change"""
        self.current_dataset = self.dataset_var.get()
        print(f"[PUZZLE BROWSER] Dataset changed to: {self.current_dataset}")
        self._load_puzzles()

    def _on_search_change(self, *args):
        """Handle search text change"""
        self._load_puzzles()

    def _on_puzzle_click(self, puzzle_id: str):
        """Handle puzzle click"""
        print(f"[PUZZLE BROWSER] Puzzle selected: {puzzle_id}")

        # Update selection highlight
        if self.selected_puzzle_id and self.selected_puzzle_id in self.thumbnails:
            self.thumbnails[self.selected_puzzle_id].highlight(False)

        self.selected_puzzle_id = puzzle_id
        if puzzle_id in self.thumbnails:
            self.thumbnails[puzzle_id].highlight(True)

        # Callback
        if self.on_puzzle_select:
            self.on_puzzle_select(puzzle_id, self.current_dataset)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def is_running(self) -> bool:
        """Check if window is still running"""
        try:
            return self.window.winfo_exists()
        except tk.TclError:
            return False

    def mainloop_iteration(self):
        """Process one iteration of the event loop (for integration with pygame)"""
        if self.is_running():
            try:
                self.window.update_idletasks()
                self.window.update()
            except tk.TclError:
                pass

    def close(self):
        """Close the window"""
        try:
            self.window.destroy()
        except:
            pass


if __name__ == "__main__":
    # Test the browser
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.data_loader import ARCDataLoader

    def on_select(puzzle_id, dataset):
        print(f"Selected: {puzzle_id} from {dataset}")

    loader = ARCDataLoader("arc-prize-2025")
    browser = PuzzleBrowser(loader, on_puzzle_select=on_select)

    browser.window.mainloop()
