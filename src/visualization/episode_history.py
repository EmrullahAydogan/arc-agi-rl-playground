#!/usr/bin/env python3
"""
Episode History Viewer - View past episodes with detailed metrics
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable
from datetime import datetime


class EpisodeHistoryViewer:
    """View and filter episode history"""

    def __init__(
        self,
        episode_recorder,
        on_replay_select: Optional[Callable] = None,
        title: str = "Episode History"
    ):
        """
        Args:
            episode_recorder: EpisodeRecorder instance
            on_replay_select: Callback when episode is selected for replay (receives episode)
            title: Window title
        """
        self.recorder = episode_recorder
        self.on_replay_select = on_replay_select
        self.title = title

        # Filter state
        self.filter_puzzle = ""
        self.filter_mode = "all"  # "all", "train", "test"
        self.filter_solved = "all"  # "all", "solved", "failed"

        # Create window
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("900x600")
        self.window.configure(bg='#1e1e1e')

        # Create UI
        self._create_ui()

        # Initial load
        self._refresh_list()

        print(f"[EPISODE HISTORY] Window created: {title}")

    def _create_ui(self):
        """Create the UI layout"""
        # Top toolbar
        toolbar = tk.Frame(self.window, bg='#2b2b2b', height=80)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Statistics panel
        stats_row = tk.Frame(toolbar, bg='#2b2b2b')
        stats_row.pack(side=tk.TOP, fill=tk.X, pady=(5, 10))

        self.stats_labels = {}
        stats_info = [
            ('Total Episodes', 'total'),
            ('Solved', 'solved'),
            ('Success Rate', 'rate'),
            ('Avg Steps', 'steps'),
            ('Avg Accuracy', 'accuracy')
        ]

        for label_text, key in stats_info:
            frame = tk.Frame(stats_row, bg='#2b2b2b')
            frame.pack(side=tk.LEFT, padx=10)

            tk.Label(
                frame,
                text=label_text + ":",
                bg='#2b2b2b',
                fg='#888888',
                font=('Arial', 9)
            ).pack(side=tk.TOP)

            label = tk.Label(
                frame,
                text="0",
                bg='#2b2b2b',
                fg='white',
                font=('Arial', 11, 'bold')
            )
            label.pack(side=tk.TOP)
            self.stats_labels[key] = label

        # Filter controls
        filter_row = tk.Frame(toolbar, bg='#2b2b2b')
        filter_row.pack(side=tk.TOP, fill=tk.X)

        # Puzzle filter
        tk.Label(
            filter_row,
            text="Puzzle ID:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(5, 5))

        self.puzzle_filter_var = tk.StringVar()
        puzzle_entry = tk.Entry(
            filter_row,
            textvariable=self.puzzle_filter_var,
            width=15,
            bg='#3c3c3c',
            fg='white',
            insertbackground='white'
        )
        puzzle_entry.pack(side=tk.LEFT, padx=5)
        self.puzzle_filter_var.trace('w', lambda *args: self._refresh_list())

        # Mode filter
        tk.Label(
            filter_row,
            text="Mode:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.mode_filter_var = tk.StringVar(value='all')
        mode_combo = ttk.Combobox(
            filter_row,
            textvariable=self.mode_filter_var,
            values=['all', 'train', 'test'],
            state='readonly',
            width=10
        )
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_list())

        # Solved filter
        tk.Label(
            filter_row,
            text="Status:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.solved_filter_var = tk.StringVar(value='all')
        solved_combo = ttk.Combobox(
            filter_row,
            textvariable=self.solved_filter_var,
            values=['all', 'solved', 'failed'],
            state='readonly',
            width=10
        )
        solved_combo.pack(side=tk.LEFT, padx=5)
        solved_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_list())

        # Refresh button
        refresh_btn = tk.Button(
            filter_row,
            text="REFRESH",
            command=self._refresh_list,
            bg='#4a4a4a',
            fg='white',
            font=('Arial', 9, 'bold'),
            padx=15,
            pady=5,
            relief=tk.FLAT
        )
        refresh_btn.pack(side=tk.LEFT, padx=(20, 5))

        # Clear button
        clear_btn = tk.Button(
            filter_row,
            text="CLEAR ALL",
            command=self._clear_all,
            bg='#aa3333',
            fg='white',
            font=('Arial', 9, 'bold'),
            padx=15,
            pady=5,
            relief=tk.FLAT
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Episode list with scrollbar
        list_frame = tk.Frame(self.window, bg='#1e1e1e')
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview for episodes
        columns = ('ID', 'Time', 'Puzzle', 'Mode', 'Sample', 'Steps', 'Reward', 'Accuracy', 'Status')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=20)

        # Column configuration
        col_widths = {
            'ID': 50,
            'Time': 80,
            'Puzzle': 120,
            'Mode': 60,
            'Sample': 60,
            'Steps': 60,
            'Reward': 80,
            'Accuracy': 80,
            'Status': 80
        }

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths[col], anchor=tk.CENTER)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind double-click for replay
        self.tree.bind('<Double-1>', self._on_episode_double_click)

        # Style configuration
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            'Treeview',
            background='#2b2b2b',
            foreground='white',
            fieldbackground='#2b2b2b',
            borderwidth=0
        )
        style.configure('Treeview.Heading', background='#3c3c3c', foreground='white')
        style.map('Treeview', background=[('selected', '#0066cc')])

    def _refresh_list(self):
        """Refresh the episode list with current filters"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Apply filters
        puzzle_filter = self.puzzle_filter_var.get().lower()
        mode_filter = self.mode_filter_var.get()
        solved_filter = self.solved_filter_var.get()

        episodes = self.recorder.episodes

        # Filter by puzzle
        if puzzle_filter:
            episodes = [e for e in episodes if puzzle_filter in e.puzzle_id.lower()]

        # Filter by mode
        if mode_filter != 'all':
            episodes = [e for e in episodes if e.mode == mode_filter]

        # Filter by solved status
        if solved_filter == 'solved':
            episodes = [e for e in episodes if e.is_solved]
        elif solved_filter == 'failed':
            episodes = [e for e in episodes if not e.is_solved]

        # Sort by episode ID (most recent first)
        episodes = sorted(episodes, key=lambda e: e.episode_id, reverse=True)

        # Add to tree
        for episode in episodes:
            time_str = episode.start_time.strftime("%H:%M:%S")
            sample_str = str(episode.sample_index + 1) if episode.mode == 'train' else '-'
            status = 'SOLVED' if episode.is_solved else 'FAILED'
            status_color = '#00ff00' if episode.is_solved else '#ff0000'

            item_id = self.tree.insert('', tk.END, values=(
                episode.episode_id,
                time_str,
                episode.puzzle_id[:12] + "..." if len(episode.puzzle_id) > 12 else episode.puzzle_id,
                episode.mode.upper(),
                sample_str,
                episode.total_steps,
                f"{episode.total_reward:.1f}",
                f"{episode.final_accuracy * 100:.1f}%",
                status
            ))

            # Color code by status
            if episode.is_solved:
                self.tree.item(item_id, tags=('solved',))
            else:
                self.tree.item(item_id, tags=('failed',))

        self.tree.tag_configure('solved', foreground='#00ff00')
        self.tree.tag_configure('failed', foreground='#ff6666')

        # Update statistics
        self._update_statistics()

    def _update_statistics(self):
        """Update statistics display"""
        stats = self.recorder.get_statistics()

        self.stats_labels['total'].config(text=str(stats['total_episodes']))
        self.stats_labels['solved'].config(text=str(stats['solved_episodes']))
        self.stats_labels['rate'].config(text=f"{stats['success_rate']:.1f}%")
        self.stats_labels['steps'].config(text=f"{stats['avg_steps']:.1f}")
        self.stats_labels['accuracy'].config(text=f"{stats['avg_accuracy']:.1f}%")

    def _on_episode_double_click(self, event):
        """Handle double-click on episode"""
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.tree.item(item, 'values')
        episode_id = int(values[0])

        episode = self.recorder.get_episode_by_id(episode_id)
        if episode and self.on_replay_select:
            print(f"[EPISODE HISTORY] Replay selected: Episode {episode_id}")
            self.on_replay_select(episode)

    def _clear_all(self):
        """Clear all episodes"""
        if messagebox.askyesno("Clear All", "Are you sure you want to clear all episode history?"):
            self.recorder.clear()
            self._refresh_list()
            print("[EPISODE HISTORY] All episodes cleared")

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

    def refresh(self):
        """Refresh the display (called externally)"""
        if self.is_running():
            self._refresh_list()

    def close(self):
        """Close the window"""
        try:
            self.window.destroy()
        except:
            pass


if __name__ == "__main__":
    # Test the viewer
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.episode_recorder import EpisodeRecorder

    # Create test data
    recorder = EpisodeRecorder()

    for i in range(20):
        episode = recorder.start_episode(
            f"puzzle_{i % 3}",
            "training",
            "train" if i % 2 == 0 else "test",
            i % 5
        )
        for step in range(10 + i % 20):
            recorder.record_step(step, 0.5, 0.3 + step * 0.05)
        recorder.end_episode(i % 3 == 0, 0.7 + i * 0.01)

    viewer = EpisodeHistoryViewer(recorder)
    viewer.window.mainloop()
