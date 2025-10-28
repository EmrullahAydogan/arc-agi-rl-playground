#!/usr/bin/env python3
"""
ARC-AGI Demonstration Playground
Human-in-the-Loop: Teach the RL agent by demonstrating puzzle solutions
"""
import sys
import argparse
from pathlib import Path
import time
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import ARCDataLoader
from src.utils.demonstration_buffer import DemonstrationBuffer, Demonstration
from src.environment.arc_env import ARCEnvironment
from src.visualization.pygame_viewer import PygameViewer


class DemoPlayground:
    """
    Demonstration Playground for Human Teaching
    Record human demonstrations for imitation learning
    """

    def __init__(
        self,
        data_dir: str = "arc-prize-2025",
        puzzle_id: str = None,
        dataset: str = "training",
        max_steps: int = 100,
        fps: int = 30,  # Higher FPS for better interactivity
        demo_save_file: str = "demonstrations.pkl"
    ):
        """
        Args:
            data_dir: ARC data directory
            puzzle_id: Specific puzzle ID (None = random)
            dataset: "training" or "evaluation"
            max_steps: Maximum steps per episode
            fps: Visualization FPS
            demo_save_file: Filename to save demonstrations
        """
        self.data_dir = data_dir
        self.dataset = dataset
        self.max_steps = max_steps
        self.fps = fps
        self.demo_save_file = demo_save_file

        # Data loader
        print("[LOADING] Loading ARC data...")
        self.loader = ARCDataLoader(data_dir)

        # Demonstration buffer
        print("[SETUP] Creating demonstration buffer...")
        self.demo_buffer = DemonstrationBuffer(save_dir="demonstrations")

        # Try to load existing demonstrations
        try:
            self.demo_buffer.load(demo_save_file)
        except:
            print("[INFO] No existing demonstrations found, starting fresh")

        # Load puzzle
        if puzzle_id:
            print(f"[LOADING] Loading puzzle: {puzzle_id}")
            puzzle_data = self.loader.get_puzzle(puzzle_id, dataset)
            self.puzzle_id = puzzle_id
        else:
            print("[LOADING] Selecting random puzzle...")
            self.puzzle_id, puzzle_data = self.loader.get_random_puzzle(dataset)
            print(f"[INFO] Selected puzzle: {self.puzzle_id}")

        if puzzle_data is None:
            raise ValueError(f"Puzzle not found: {puzzle_id}")

        # Create environment
        print("[SETUP] Creating environment...")
        self.env = ARCEnvironment(
            puzzle_data=puzzle_data,
            task_index=0,
            max_steps=max_steps
        )

        # Visualizer (with high FPS for interactivity)
        print("[SETUP] Creating visualization interface...")
        self.viewer = PygameViewer(fps=fps)

        # Current demonstration tracking
        self.current_demo = None
        self.demo_start_time = None
        self.is_recording = False

        print("[OK] Demonstration playground ready!")
        print("[INFO] EDIT MODE is your main tool for teaching!\n")

    def run(self):
        """Main demonstration loop"""
        print("=" * 80)
        print(">>> ARC-AGI DEMONSTRATION PLAYGROUND <<<")
        print("="* 80)
        print(f"[INFO] Puzzle ID: {self.puzzle_id}")
        print(f"[INFO] Max Steps: {self.max_steps}")
        print(f"[INFO] FPS: {self.fps}")
        print(f"[INFO] Save File: {self.demo_save_file}")
        print("\n>>> DEMONSTRATION MODE:")
        print("   This is YOUR chance to teach the AI how to solve ARC puzzles!")
        print("\n>>> HOW IT WORKS:")
        print("   1. Press 'E' or click 'EDIT MODE' button to start teaching")
        print("   2. Select a color from the palette (left side)")
        print("   3. Click on CURRENT grid to paint cells")
        print("   4. Recording happens automatically when you make changes")
        print("   5. Press SPACE to start/pause")
        print("   6. Press 'R' to reset if you make a mistake")
        print("   7. When done, the demonstration is saved automatically!")
        print("\n>>> CONTROLS:")
        print("   E: Toggle Edit Mode (REQUIRED for teaching)")
        print("   SPACE: Pause/Resume")
        print("   R: Reset current puzzle")
        print("   T: Switch mode (Train <-> Test)")
        print("   N: Next training sample")
        print("   LEFT ARROW: Previous sample")
        print("   Q/ESC: Quit and save demonstrations")
        print("\n>>> STATISTICS:")
        self.demo_buffer.print_summary()
        print("=" * 80)
        print()

        # Initial reset
        observation, info = self.env.reset()

        # Track previous grid state to detect human edits
        previous_grid = self.env.current_grid.copy()

        running = True
        episode_count = 0
        step_count = 0

        # Start first demonstration
        self._start_demonstration()

        while running:
            # Event handling
            controls = self.viewer.handle_events(current_grid=self.env.current_grid)

            if controls['quit']:
                print("\n[EXIT] Saving demonstrations and exiting...")
                self._end_demonstration()
                self.demo_buffer.save(self.demo_save_file)
                running = False
                break

            if controls['reset']:
                print(f"\n[RESET] Resetting episode {episode_count}...")
                # End current demo if it has steps
                if self.current_demo and self.current_demo.total_steps > 0:
                    self._end_demonstration()

                # Reset environment
                observation, info = self.env.reset()
                previous_grid = self.env.current_grid.copy()
                step_count = 0

                # Start new demonstration
                self._start_demonstration()

                self.viewer.reset_control_flags()
                self.viewer.reset_heatmap()
                continue

            # Mode toggle
            if controls.get('toggle_mode', False):
                new_mode = self.env.switch_mode()
                mode_str = "TRAIN" if new_mode == 'train' else "TEST"
                state_info = self.env.get_state_info()
                sample_info = ""
                if new_mode == 'train':
                    sample_info = f" (Sample {state_info['train_sample_index'] + 1}/{state_info['num_train_samples']})"
                print(f"\n[MODE] Mode changed -> {mode_str}{sample_info}")

                # End current demo
                if self.current_demo and self.current_demo.total_steps > 0:
                    self._end_demonstration()

                observation, info = self.env.reset()
                previous_grid = self.env.current_grid.copy()

                # Start new demo
                self._start_demonstration()

                self.viewer.reset_control_flags()
                continue

            # Next sample
            if controls.get('next_sample', False):
                if self.env.mode == 'train':
                    # End current demo
                    if self.current_demo and self.current_demo.total_steps > 0:
                        self._end_demonstration()

                    new_index = self.env.next_sample()
                    state_info = self.env.get_state_info()
                    print(f"\n[NEXT] Train Sample: {new_index + 1}/{state_info['num_train_samples']}")
                    observation, info = self.env.reset()
                    previous_grid = self.env.current_grid.copy()

                    # Start new demo
                    self._start_demonstration()

                    self.viewer.reset_control_flags()
                    continue

            # Previous sample
            if controls.get('prev_sample', False):
                if self.env.mode == 'train':
                    # End current demo
                    if self.current_demo and self.current_demo.total_steps > 0:
                        self._end_demonstration()

                    new_index = self.env.previous_sample()
                    state_info = self.env.get_state_info()
                    print(f"\n[PREV] Train Sample: {new_index + 1}/{state_info['num_train_samples']}")
                    observation, info = self.env.reset()
                    previous_grid = self.env.current_grid.copy()

                    # Start new demo
                    self._start_demonstration()

                    self.viewer.reset_control_flags()
                    continue

            # Detect human edits to the grid
            if not np.array_equal(self.env.current_grid, previous_grid):
                # Human made an edit!
                action = self._detect_human_action(previous_grid, self.env.current_grid)

                if action is not None:
                    # Execute action in environment to get reward
                    next_observation, reward, terminated, truncated, step_info = self.env.step(action)

                    # Record this step in demonstration
                    state_info = self.env.get_state_info()

                    # Calculate accuracy
                    target_height, target_width = state_info['target_grid'].shape
                    current_slice = state_info['current_grid'][:target_height, :target_width]
                    correct_pixels = np.sum(current_slice == state_info['target_grid'])
                    total_pixels = target_height * target_width
                    accuracy = correct_pixels / total_pixels

                    # Add step to demonstration
                    self.current_demo.add_step(
                        state=previous_grid,
                        action=action,
                        reward=reward,
                        info=step_info
                    )

                    step_count += 1

                    # Record for heatmap
                    if 'action_decoded' in step_info:
                        self.viewer.record_action(step_info['action_decoded'])

                    print(f"[DEMO] Step {step_count}: Action {action}, Reward: {reward:.2f}, Accuracy: {accuracy*100:.1f}%")

                    # Check if solved
                    if terminated:
                        print(f"\n[SUCCESS] Puzzle SOLVED!")
                        print(f"   Total Steps: {step_count}")
                        print(f"   Final Accuracy: {accuracy * 100:.1f}%")

                        # End demonstration
                        self._end_demonstration(solved=True, final_accuracy=accuracy*100)

                        # Auto-save
                        self.demo_buffer.save(self.demo_save_file)

                        # Wait a bit
                        time.sleep(2)

                        # Reset for new episode
                        episode_count += 1
                        print(f"\n[NEW EPISODE] Starting episode {episode_count + 1}...")
                        observation, info = self.env.reset()
                        previous_grid = self.env.current_grid.copy()
                        step_count = 0

                        # Start new demo
                        self._start_demonstration()

                    # Update previous grid
                    previous_grid = self.env.current_grid.copy()
                    observation = next_observation

            # Render
            state_info = self.env.get_state_info()
            self._render(state_info)

            # Reset control flags
            self.viewer.reset_control_flags()

        # Cleanup
        self.viewer.close()
        print("\n[OK] Demonstration playground closed!")

    def _start_demonstration(self):
        """Start recording a new demonstration"""
        state_info = self.env.get_state_info()

        self.current_demo = Demonstration(
            puzzle_id=self.puzzle_id,
            mode=state_info['mode'],
            sample_idx=state_info['train_sample_index']
        )

        self.demo_start_time = time.time()
        self.is_recording = True

        print(f"\n[DEMO START] Recording demonstration for {self.puzzle_id}")
        print(f"   Mode: {state_info['mode']}")
        if state_info['mode'] == 'train':
            print(f"   Sample: {state_info['train_sample_index'] + 1}/{state_info['num_train_samples']}")

    def _end_demonstration(self, solved: bool = False, final_accuracy: float = 0.0):
        """End current demonstration and save to buffer"""
        if self.current_demo is None:
            return

        if self.current_demo.total_steps == 0:
            print("[DEMO] No steps recorded, skipping save")
            self.current_demo = None
            return

        # Finalize demo
        duration = time.time() - self.demo_start_time
        self.current_demo.finalize(solved, final_accuracy, duration)

        # Add to buffer
        self.demo_buffer.add_demonstration(self.current_demo)

        # Reset
        self.current_demo = None
        self.is_recording = False

        print(f"[DEMO END] Demonstration saved to buffer")

    def _detect_human_action(self, previous_grid: np.ndarray, current_grid: np.ndarray) -> int:
        """
        Detect what action human performed by comparing grids

        Args:
            previous_grid: Grid before edit
            current_grid: Grid after edit

        Returns:
            action: Action ID that corresponds to the change
        """
        # Find changed cells
        diff_mask = previous_grid != current_grid
        changed_indices = np.argwhere(diff_mask)

        if len(changed_indices) == 0:
            return None

        # Take the first changed cell (assume human changes one cell at a time)
        x, y = changed_indices[0]
        new_color = int(current_grid[x, y])

        # Convert to action
        # Action = x * (30 * 10) + y * 10 + color
        action = x * (30 * 10) + y * 10 + new_color

        return action

    def _render(self, state_info: dict):
        """Render current state"""
        viewer_info = {
            'puzzle_id': self.puzzle_id,
            'dataset': self.dataset,
            'mode': state_info['mode'],
            'train_sample_index': state_info['train_sample_index'],
            'num_train_samples': state_info['num_train_samples'],
            'num_test_samples': state_info['num_test_samples'],
            'steps': state_info['steps'],
            'max_steps': state_info['max_steps'],
            'total_reward': state_info['total_reward'],
            'last_reward': state_info['last_reward'],
            'is_solved': state_info['is_solved'],
            'done': state_info['done'],
            'agent_type': 'HUMAN (Teacher)',
            'agent_actions': self.current_demo.total_steps if self.current_demo else 0,
            'agent_params': {
                'mode': 'Demonstration Recording',
                'total_demos': self.demo_buffer.total_demonstrations,
                'total_steps_recorded': self.demo_buffer.total_steps,
                'solved_demos': self.demo_buffer.solved_count,
            }
        }

        if state_info['last_action'] is not None:
            x, y, color = self.env._action_to_grid_operation(state_info['last_action'])
            viewer_info['last_action_decoded'] = (x, y, color)

        # Render Pygame window
        self.viewer.render(
            input_grid=state_info['input_grid'],
            current_grid=state_info['current_grid'],
            target_grid=state_info['target_grid'],
            info=viewer_info
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ARC-AGI Demonstration Playground")
    parser.add_argument(
        '--puzzle-id',
        type=str,
        default=None,
        help='Puzzle ID (blank = random)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='training',
        choices=['training', 'evaluation'],
        help='Dataset selection'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Visualization FPS (higher = more responsive)'
    )
    parser.add_argument(
        '--demo-file',
        type=str,
        default='demonstrations.pkl',
        help='Demonstration save file'
    )

    args = parser.parse_args()

    # Create and run playground
    try:
        playground = DemoPlayground(
            puzzle_id=args.puzzle_id,
            dataset=args.dataset,
            max_steps=args.max_steps,
            fps=args.fps,
            demo_save_file=args.demo_file
        )
        playground.run()
    except KeyboardInterrupt:
        print("\n\n[WARNING] KeyboardInterrupt - Exiting...")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
