#!/usr/bin/env python3
"""
Replay Viewer - Playback recorded episodes with step-by-step controls
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable


class ReplayViewer:
    """Control panel for replaying recorded episodes"""

    def __init__(
        self,
        episode,
        on_step_change: Optional[Callable] = None,
        title: str = "Replay Control"
    ):
        """
        Args:
            episode: Episode object to replay
            on_step_change: Callback when step changes (receives step_index, grid_state, action, reward, accuracy)
            title: Window title
        """
        self.episode = episode
        self.on_step_change = on_step_change
        self.title = title

        # Replay state
        self.current_step = 0
        self.is_playing = False
        self.playback_speed = 1.0  # Steps per second
        self.last_update_time = 0

        # Create window
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("600x350")
        self.window.configure(bg='#1e1e1e')

        # Create UI
        self._create_ui()

        # Initial state
        self._update_display()
        self._trigger_step_change()

        print(f"[REPLAY] Viewer created for Episode {episode.episode_id}")

    def _create_ui(self):
        """Create the UI layout"""
        # Episode info header
        header = tk.Frame(self.window, bg='#2b2b2b', height=100)
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Episode title
        title_text = f"Episode {self.episode.episode_id} - {self.episode.puzzle_id}"
        tk.Label(
            header,
            text=title_text,
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 14, 'bold')
        ).pack(side=tk.TOP, pady=(5, 10))

        # Episode details row
        details_frame = tk.Frame(header, bg='#2b2b2b')
        details_frame.pack(side=tk.TOP, fill=tk.X)

        details = [
            ('Mode', self.episode.mode.upper()),
            ('Sample', str(self.episode.sample_index + 1) if self.episode.mode == 'train' else 'Test'),
            ('Total Steps', str(self.episode.total_steps)),
            ('Status', 'SOLVED' if self.episode.is_solved else 'FAILED'),
            ('Accuracy', f"{self.episode.final_accuracy * 100:.1f}%")
        ]

        for label, value in details:
            frame = tk.Frame(details_frame, bg='#2b2b2b')
            frame.pack(side=tk.LEFT, padx=15)

            tk.Label(
                frame,
                text=label + ":",
                bg='#2b2b2b',
                fg='#888888',
                font=('Arial', 9)
            ).pack(side=tk.TOP)

            color = '#00ff00' if label == 'Status' and self.episode.is_solved else 'white'
            tk.Label(
                frame,
                text=value,
                bg='#2b2b2b',
                fg=color,
                font=('Arial', 10, 'bold')
            ).pack(side=tk.TOP)

        # Progress section
        progress_frame = tk.Frame(self.window, bg='#1e1e1e')
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(10, 5))

        # Current step display
        step_info_frame = tk.Frame(progress_frame, bg='#1e1e1e')
        step_info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.step_label = tk.Label(
            step_info_frame,
            text=f"Step: 0 / {self.episode.total_steps}",
            bg='#1e1e1e',
            fg='white',
            font=('Arial', 11)
        )
        self.step_label.pack(side=tk.LEFT)

        # Current metrics display
        self.metrics_label = tk.Label(
            step_info_frame,
            text="",
            bg='#1e1e1e',
            fg='#aaaaaa',
            font=('Arial', 10)
        )
        self.metrics_label.pack(side=tk.RIGHT)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=560,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.TOP, fill=tk.X)
        self.progress_bar['maximum'] = max(1, self.episode.total_steps)
        self.progress_bar['value'] = 0

        # Playback controls
        controls_frame = tk.Frame(self.window, bg='#2b2b2b', height=80)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=15)

        # Control buttons row
        buttons_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        buttons_frame.pack(side=tk.TOP, pady=(10, 15))

        # Step backward button
        self.step_back_btn = tk.Button(
            buttons_frame,
            text="< STEP",
            command=self._step_backward,
            bg='#4a4a4a',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=8,
            pady=8,
            relief=tk.FLAT
        )
        self.step_back_btn.pack(side=tk.LEFT, padx=5)

        # Play/Pause button
        self.play_pause_btn = tk.Button(
            buttons_frame,
            text="PLAY",
            command=self._toggle_play,
            bg='#00aa00',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=10,
            pady=8,
            relief=tk.FLAT
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        # Step forward button
        self.step_forward_btn = tk.Button(
            buttons_frame,
            text="STEP >",
            command=self._step_forward,
            bg='#4a4a4a',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=8,
            pady=8,
            relief=tk.FLAT
        )
        self.step_forward_btn.pack(side=tk.LEFT, padx=5)

        # Reset button
        self.reset_btn = tk.Button(
            buttons_frame,
            text="RESET",
            command=self._reset,
            bg='#aa3333',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=8,
            pady=8,
            relief=tk.FLAT
        )
        self.reset_btn.pack(side=tk.LEFT, padx=(20, 5))

        # Speed control row
        speed_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        speed_frame.pack(side=tk.TOP)

        tk.Label(
            speed_frame,
            text="Speed:",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Speed buttons
        speeds = [('0.5x', 0.5), ('1x', 1.0), ('2x', 2.0), ('4x', 4.0)]
        self.speed_buttons = []

        for label, speed in speeds:
            btn = tk.Button(
                speed_frame,
                text=label,
                command=lambda s=speed: self._set_speed(s),
                bg='#4a4a4a' if speed != 1.0 else '#00aa00',
                fg='white',
                font=('Arial', 9),
                width=5,
                pady=4,
                relief=tk.FLAT
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.speed_buttons.append((btn, speed))

        # Slider for step navigation
        slider_frame = tk.Frame(self.window, bg='#1e1e1e')
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(5, 10))

        tk.Label(
            slider_frame,
            text="Jump to step:",
            bg='#1e1e1e',
            fg='white',
            font=('Arial', 9)
        ).pack(side=tk.TOP, anchor=tk.W)

        self.step_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=max(1, self.episode.total_steps),
            orient=tk.HORIZONTAL,
            bg='#2b2b2b',
            fg='white',
            troughcolor='#1e1e1e',
            highlightthickness=0,
            command=self._on_slider_change
        )
        self.step_slider.pack(side=tk.TOP, fill=tk.X)

    def _update_display(self):
        """Update all display elements"""
        # Update step label
        self.step_label.config(text=f"Step: {self.current_step} / {self.episode.total_steps}")

        # Update progress bar
        self.progress_bar['value'] = self.current_step

        # Update slider
        self.step_slider.set(self.current_step)

        # Update metrics
        if self.current_step > 0 and self.current_step <= len(self.episode.rewards):
            reward = self.episode.rewards[self.current_step - 1]
            accuracy = self.episode.accuracies[self.current_step - 1]
            self.metrics_label.config(
                text=f"Reward: {reward:.2f}  |  Accuracy: {accuracy * 100:.1f}%"
            )
        else:
            self.metrics_label.config(text="Initial state")

    def _trigger_step_change(self):
        """Trigger the step change callback"""
        if self.on_step_change:
            # Get current state
            if self.current_step == 0:
                # Initial state
                grid_state = None
                action = None
                reward = 0.0
                accuracy = 0.0
            else:
                idx = self.current_step - 1
                grid_state = self.episode.grids[idx] if idx < len(self.episode.grids) else None
                action = self.episode.actions[idx] if idx < len(self.episode.actions) else None
                reward = self.episode.rewards[idx] if idx < len(self.episode.rewards) else 0.0
                accuracy = self.episode.accuracies[idx] if idx < len(self.episode.accuracies) else 0.0

            self.on_step_change(self.current_step, grid_state, action, reward, accuracy)

    def _step_forward(self):
        """Step forward one action"""
        if self.current_step < self.episode.total_steps:
            self.current_step += 1
            self._update_display()
            self._trigger_step_change()

    def _step_backward(self):
        """Step backward one action"""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_display()
            self._trigger_step_change()

    def _toggle_play(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_pause_btn.config(text="PAUSE", bg='#aa6600')
            import time
            self.last_update_time = time.time()
        else:
            self.play_pause_btn.config(text="PLAY", bg='#00aa00')

    def _reset(self):
        """Reset to beginning"""
        self.current_step = 0
        self.is_playing = False
        self.play_pause_btn.config(text="PLAY", bg='#00aa00')
        self._update_display()
        self._trigger_step_change()

    def _set_speed(self, speed: float):
        """Set playback speed"""
        self.playback_speed = speed

        # Update button colors
        for btn, btn_speed in self.speed_buttons:
            if btn_speed == speed:
                btn.config(bg='#00aa00')
            else:
                btn.config(bg='#4a4a4a')

    def _on_slider_change(self, value):
        """Handle slider value change"""
        new_step = int(float(value))
        if new_step != self.current_step:
            self.current_step = new_step
            self._update_display()
            self._trigger_step_change()

    def update(self):
        """Update playback (call from main loop)"""
        if self.is_playing:
            import time
            current_time = time.time()
            elapsed = current_time - self.last_update_time

            # Check if enough time has passed for next step
            step_interval = 1.0 / self.playback_speed
            if elapsed >= step_interval:
                self._step_forward()
                self.last_update_time = current_time

                # Stop at end
                if self.current_step >= self.episode.total_steps:
                    self.is_playing = False
                    self.play_pause_btn.config(text="PLAY", bg='#00aa00')

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
                self.update()  # Update playback
            except tk.TclError:
                pass

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

    from src.utils.episode_recorder import Episode

    # Create test episode
    episode = Episode(1, "test_puzzle", "training", "train", 0)
    for i in range(20):
        episode.record_step(i, 0.5 + i * 0.02, 0.3 + i * 0.03, [[0]*5 for _ in range(5)])
    episode.end_episode(True, 0.85)

    def on_step(step, grid, action, reward, accuracy):
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Acc={accuracy:.2f}")

    viewer = ReplayViewer(episode, on_step_change=on_step)
    viewer.window.mainloop()
