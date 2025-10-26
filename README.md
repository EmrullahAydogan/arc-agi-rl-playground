# ARC-AGI Reinforcement Learning Playground

Interactive training environment for reinforcement learning agents on ARC-AGI puzzles with real-time visualization, comprehensive episode tracking, and playback controls.

## ✨ Features

### Multi-Window System
The playground features **five independent windows** for a comprehensive training and analysis experience:

1. **Main Window (Pygame)** - Real-time grid visualization
2. **Info Panel (Tkinter)** - Live metrics and agent parameters
3. **Puzzle Browser (Tkinter)** - Browse and select from 400+ puzzles
4. **Episode History (Tkinter)** - Track and filter past episodes
5. **Replay Viewer (Tkinter)** - Step-by-step episode playback

All windows support multi-monitor setups and can be positioned independently!

### Main Window (Pygame)
- **Full-screen grid visualization**: INPUT, CURRENT, and TARGET grids side by side
- **Dynamic layouts**: Switch between horizontal (side-by-side) and vertical (stacked) layouts
- **Smart zoom system**:
  - Auto-fit mode: Automatically scales grids to fit window (8-40px range)
  - Manual zoom: Mouse wheel or Z/X keys
  - Adaptive: Automatically chooses best cell size for current grid
- **Manual grid editing**: Click to paint cells with selected colors
- **Heatmap overlay**: Visualize agent activity across the grid
- **Color palette**: 10 ARC colors for manual editing
- **Resizable window**: Minimum 900x700, scales to any size
- **Button controls**: Clickable buttons for all major functions
- **Top-right controls**: Quick access to HISTORY and BROWSE PUZZLES

### Info Panel (Tkinter)
- **Status information**: Real-time episode stats, steps, rewards
- **Agent parameters**: Learning rate, epsilon, gamma, batch size, etc.
- **Performance metrics**:
  - Train mode: Episodes, success rate, average accuracy
  - Test mode: Test performance tracking
- **Sample status**: Track which training samples are solved with color coding
  - Green [OK]: Sample solved
  - Yellow [~]: Partial progress (>50% accuracy)
  - Red [X]: Failed attempts (<50% accuracy)
  - Gray [-]: Not yet attempted
- **Last action display**: See the agent's most recent move

### Puzzle Browser (NEW!)
Browse and switch between puzzles instantly:
- **Grid layout**: 5 columns of puzzle thumbnails
- **Visual previews**: First training sample rendered for each puzzle
- **Puzzle information**: ID, grid size, train/test sample counts
- **Dataset selector**: Switch between Training (400 puzzles) and Evaluation (400 puzzles)
- **Search functionality**: Filter puzzles by ID
- **Interactive selection**: Click any puzzle to load it instantly
- **Hover effects**: Visual feedback on mouse hover
- **Scrollable**: Smooth scrolling through all puzzles

### Episode History (NEW!)
Comprehensive tracking of all training episodes:
- **Detailed table**: 9 columns (ID, Time, Puzzle, Mode, Sample, Steps, Reward, Accuracy, Status)
- **Live statistics**:
  - Total episodes and solved count
  - Success rate percentage
  - Average steps and accuracy
- **Advanced filtering**:
  - By puzzle ID (search)
  - By mode (all/train/test)
  - By status (all/solved/failed)
- **Color-coded display**:
  - Green: SOLVED episodes
  - Red: FAILED episodes
- **Episode management**:
  - Auto-refresh on episode completion
  - Manual refresh button
  - Clear all with confirmation
- **Replay integration**: Double-click any episode to replay it

### Replay Viewer (NEW!)
Watch recorded episodes step-by-step:
- **Episode information header**:
  - Episode ID, Puzzle ID
  - Mode, Sample, Total Steps
  - Status (SOLVED/FAILED) with color coding
  - Final accuracy percentage
- **Progress tracking**:
  - Current step display
  - Progress bar
  - Current reward and accuracy
- **Playback controls**:
  - Play/Pause with auto-progression
  - Step Forward/Backward for manual control
  - Reset to beginning
  - Step slider for instant jumping
- **Speed control**: 0.5x, 1x, 2x, 4x playback speeds
- **Visual sync**: Grid states update in main window
- **Auto-pause**: Stops at end of episode

### Train/Test Mode System
- **Train Mode**: Learn from multiple training examples
  - Navigate between training samples with PREV/NEXT buttons
  - See input-output pairs for supervised learning
  - Track per-sample performance metrics
- **Test Mode**: Evaluate on the test puzzle
  - Agent predicts the output for the test input
  - Test-specific metrics tracking
- **Easy switching**: Toggle between modes with MODE button

### Performance Tracking
- **Episode metrics**: Track success rate, accuracy, rewards
- **Per-sample tracking**: See which training samples are mastered
- **Real-time updates**: Metrics update as the agent trains
- **Historical data**: Keep track of all attempts and improvements (500+ episodes)
- **Persistent storage**: Save/load episode history to JSON files

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/EmrullahAydogan/arc-agi-rl-playground.git
cd arc-agi-rl-playground
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download ARC Data

Download the ARC dataset and place it in the `arc-prize-2025/` directory:
- `arc-prize-2025/arc-agi_training_challenges.json`
- `arc-prize-2025/arc-agi_training_solutions.json`
- `arc-prize-2025/arc-agi_evaluation_challenges.json`
- `arc-prize-2025/arc-agi_evaluation_solutions.json`

## Usage

### Basic Usage

Start with a random puzzle:
```bash
python playground.py
```

Start with a specific puzzle:
```bash
python playground.py --puzzle-id 00576224
```

### Command Line Arguments

```bash
python playground.py --help
```

Available options:
- `--puzzle-id`: Specific puzzle ID (random if not specified)
- `--dataset`: `training` or `evaluation` (default: training)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--fps`: Visualization speed (default: 5)
- `--seed`: Random seed for reproducibility

### Controls

#### Main Control Buttons (Bottom Panel - Row 1)

| Button | Function |
|--------|----------|
| `> START` / `\|\| PAUSE` | Start/pause the agent |
| `RESET` | Reset current episode |
| `<> MODE` | Switch between Train/Test modes |
| `< PREV` / `NEXT >` | Navigate training samples |
| `SPEED -` / `SPEED +` | Adjust visualization speed |
| `LAYOUT` | Toggle horizontal/vertical layout |

#### Feature Buttons (Bottom Panel - Row 2)

| Button | Function |
|--------|----------|
| `EDIT: OFF/ON` | Toggle manual grid editing mode |
| `HEATMAP: OFF/ON` | Toggle agent activity heatmap overlay |

#### Top-Right Buttons

| Button | Function |
|--------|----------|
| `HISTORY` | Open episode history viewer |
| `BROWSE PUZZLES` | Open puzzle browser |

#### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `SPACE` | Pause/Resume |
| `→` | Single step (when paused) |
| `R` | Reset episode |
| `Q` / `ESC` | Quit |
| `Z` | Zoom in |
| `X` | Zoom out |
| `A` | Toggle auto-fit mode |
| `E` | Toggle edit mode (alternative to button) |
| `H` | Toggle heatmap (alternative to button) |
| `Mouse Wheel` | Zoom in/out |

### Workflows

#### Basic Training Workflow

1. **Start in Train Mode**: System starts paused in train mode
2. **Click START**: Agent begins attempting the first training sample
3. **Navigate Samples**: Use PREV/NEXT to try different training examples
4. **Monitor Progress**: Watch metrics in the info panel
5. **Switch to Test**: Click MODE button to evaluate on test puzzle
6. **Iterate**: Return to train mode to improve the agent

#### Puzzle Browsing Workflow

1. **Click BROWSE PUZZLES** (top-right): Opens puzzle browser window
2. **Select Dataset**: Choose Training or Evaluation from dropdown
3. **Search (Optional)**: Filter puzzles by ID in search box
4. **Browse**: Scroll through puzzle thumbnails
5. **Select**: Click any puzzle to load it instantly
6. **Continue Training**: Agent starts fresh on the new puzzle

#### Episode Review Workflow

1. **Run Episodes**: Let the agent train for several episodes
2. **Click HISTORY** (top-right): Opens episode history viewer
3. **Review Statistics**: Check success rate, avg steps, accuracy
4. **Filter Episodes**: Use filters to find specific episodes
5. **Analyze Performance**: Sort and review episode details
6. **Replay Episode**: Double-click an episode to watch replay

#### Episode Replay Workflow

1. **Open History**: Click HISTORY button
2. **Select Episode**: Double-click any episode in the table
3. **Replay Opens**: Replay viewer window appears with controls
4. **Control Playback**:
   - Click PLAY for auto-playback
   - Use STEP buttons for manual control
   - Adjust speed (0.5x to 4x)
   - Use slider to jump to any step
5. **Watch in Main Window**: Grid updates show each step
6. **Close When Done**: Replay viewer closes, returns to normal mode

#### Manual Editing Workflow

1. **Enable Edit Mode**: Click EDIT button (or press E)
2. **Select Color**: Click a color in the color palette (left side)
3. **Paint Grid**: Click CURRENT grid cells to change colors
4. **Test Solution**: Run agent or manually test your changes
5. **Disable Edit Mode**: Click EDIT button again (or press E)

### Understanding the Display

**Main Window (Pygame):**
- **INPUT**: The starting grid for the current sample
- **CURRENT**: The agent's working grid (highlighted in yellow)
- **TARGET**: The goal grid to match
- **Color Palette** (left side): 10 ARC colors for manual editing
- **Control Buttons** (bottom): Two rows of control buttons
- **Top-Right Buttons**: HISTORY and BROWSE PUZZLES

**Info Panel (Tkinter):**
- **STATUS**: Current episode information
- **AGENT PARAMETERS**: Agent hyperparameters
- **PERFORMANCE METRICS**: Training and testing statistics
- **SAMPLE STATUS**: Which training samples are solved

**Puzzle Browser (Tkinter):**
- **Toolbar**: Dataset selector, search box, puzzle count
- **Grid Area**: Scrollable grid of puzzle thumbnails
- **Thumbnail Info**: Puzzle ID, grid size, sample counts

**Episode History (Tkinter):**
- **Statistics Panel**: Overall performance metrics
- **Filter Controls**: Puzzle ID search, mode/status filters
- **Episode Table**: Detailed list of all episodes
- **Action Buttons**: Refresh and clear all

**Replay Viewer (Tkinter):**
- **Episode Header**: Basic episode information
- **Progress Display**: Current step, metrics, progress bar
- **Control Buttons**: Play/Pause, Step, Reset
- **Speed Controls**: 0.5x, 1x, 2x, 4x buttons
- **Step Slider**: Jump to any step instantly

**Header Information:**
- Puzzle ID and dataset
- Current mode (TRAIN/TEST)
- Sample number (in train mode)
- Sample accuracy and solved status (color-coded)

## Project Structure

```
arc_prize_2025_RL/
├── src/
│   ├── environment/
│   │   └── arc_env.py                # Gymnasium-compatible ARC environment
│   ├── visualization/
│   │   ├── pygame_viewer.py          # Main visualization window
│   │   ├── info_panel.py             # Metrics/info panel (Tkinter)
│   │   ├── puzzle_browser.py         # Puzzle browser window (NEW)
│   │   ├── episode_history.py        # Episode history viewer (NEW)
│   │   └── replay_viewer.py          # Replay control panel (NEW)
│   ├── agents/
│   │   ├── base_agent.py             # Agent interface
│   │   └── random_agent.py           # Example random agent
│   └── utils/
│       ├── data_loader.py            # ARC data loader
│       ├── metrics_tracker.py        # Performance tracking
│       └── episode_recorder.py       # Episode recording (NEW)
├── arc-prize-2025/                   # ARC dataset files
├── playground.py                     # Main entry point
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## Creating Your Own Agent

### 1. Create Agent Class

Create a new file in `src/agents/`:

```python
from src.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, action_space, observation_space, **kwargs):
        super().__init__(action_space, observation_space)
        # Initialize your agent

    def select_action(self, observation, info=None):
        # Your action selection logic
        action = ...
        return action

    def update(self, observation, action, reward, next_observation, done, info=None):
        # Your learning logic
        pass

    def get_params(self):
        # Return hyperparameters for display
        return {
            'learning_rate': 0.001,
            'epsilon': 0.1,
            # ... other params
        }
```

### 2. Use Your Agent

Modify `playground.py`:

```python
from src.agents.my_agent import MyAgent

# In ARCPlayground.__init__():
self.agent = MyAgent(
    action_space=self.env.action_space,
    observation_space=self.env.observation_space,
    seed=seed
)
```

## Environment API

### Gymnasium Interface

The `ARCEnvironment` follows the Gymnasium API:

```python
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
```

### Spaces

**Action Space:**
- Type: `Discrete(max_grid_size * max_grid_size * num_colors)`
- Each action represents placing a color at a specific (x, y) position
- Actions are encoded as: `action = x * max_grid_size * num_colors + y * num_colors + color`

**Observation Space:**
- Type: `Box(0, 9, shape=(max_grid_size, max_grid_size))`
- Grid with values 0-9 representing colors

### Rewards

- **Dense reward**: `accuracy * 10.0` (where accuracy = correct_pixels / total_pixels)
- **Sparse reward**: `100.0` when puzzle is completely solved
- Customizable in `arc_env.py`

### Mode System

The environment supports two modes:

**Train Mode:**
- Access to multiple input-output example pairs
- Agent learns from these examples
- Navigate between samples with `env.next_sample()` and `env.previous_sample()`

**Test Mode:**
- Single test puzzle to solve
- Output not provided (agent must predict)
- Evaluation mode for trained agents

Switch modes: `env.switch_mode()`

## Episode Recording & Replay

### Recording Episodes

Episodes are automatically recorded with full history:

```python
# Episode recording happens automatically
# Includes: actions, rewards, accuracies, grid states

# Access recorded episodes
episodes = playground.episode_recorder.get_episodes()
solved_episodes = playground.episode_recorder.get_episodes(solved_only=True)

# Get episode by ID
episode = playground.episode_recorder.get_episode_by_id(42)

# Save/load to file
playground.episode_recorder.save_to_file("episodes.json")
playground.episode_recorder.load_from_file("episodes.json")
```

### Replaying Episodes

Replay any recorded episode:

1. Through UI: Double-click episode in Episode History
2. Programmatically: Call `playground._on_replay_selected(episode)`

The replay viewer provides:
- Step-by-step playback
- Variable speed control
- Jump to any step
- Visual sync with main window

## Metrics Tracking

The `MetricsTracker` class provides comprehensive performance monitoring:

```python
# Start episode
metrics.start_episode(mode='train', sample_index=0)

# Record each step
metrics.record_step(reward, accuracy)

# End episode
metrics.end_episode(is_solved=True, final_accuracy=0.95)

# Get metrics
train_metrics = metrics.get_train_metrics()
test_metrics = metrics.get_test_metrics()
sample_metrics = metrics.get_sample_metrics(sample_index=0)
```

## Development Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Gymnasium environment
- [x] Dual-window visualization
- [x] Train/Test mode system
- [x] Basic metrics tracking
- [x] Agent interface

### Phase 2: UI/UX Enhancements ✅
- [x] Puzzle Browser with thumbnails
- [x] Episode History with filtering
- [x] Replay System with playback controls
- [x] Manual grid editing
- [x] Heatmap overlay
- [x] Color palette
- [x] Button-based controls

### Phase 3: Agent Development (Next)
- [ ] DQN agent implementation
- [ ] PPO agent implementation
- [ ] A2C agent implementation
- [ ] Custom neural network architectures
- [ ] Hyperparameter tuning interface

### Phase 4: Advanced Features (Future)
- [ ] Multi-puzzle training
- [ ] Curriculum learning
- [ ] Action space extensions (transformations, patterns)
- [ ] Custom reward shaping
- [ ] Model checkpointing and loading
- [ ] Experiment tracking integration
- [ ] Multi-agent comparison
- [ ] Distributed training support

## Tips & Best Practices

### For Training
- Start with simple puzzles to verify your agent works
- Use train mode to learn patterns from examples
- Monitor per-sample metrics to identify difficult patterns
- Use episode history to track improvement over time
- Replay failed episodes to debug agent behavior

### For Debugging
- Use edit mode to test manual solutions
- Enable heatmap to see where agent focuses
- Step through episodes manually (→ key when paused)
- Review episode history for patterns in failures
- Compare successful vs failed episodes

### For Performance
- Lower FPS for faster training (--fps 2)
- Disable auto-fit and use fixed zoom
- Run multiple instances for parallel training
- Use evaluation dataset for final testing
- Save episode history regularly

## Screenshots

The playground features:
- Real-time grid visualization with 3 grids side-by-side
- Live performance metrics in separate info panel
- Interactive puzzle browser with 400+ puzzles
- Comprehensive episode history with statistics
- Replay viewer with step-by-step controls
- Multi-monitor support for all windows
- Clean, professional UI with no rendering issues

## Troubleshooting

**Grid dimensions mismatch error:**
- Fixed in latest version with adaptive grid slicing

**Unicode symbols not displaying:**
- All unicode replaced with ASCII for better compatibility

**Windows not appearing:**
- Check if pygame and tkinter are properly installed
- Verify display is available (not SSH without X11)
- Try `sudo apt-get install python3-tk` on Linux

**Performance issues:**
- Reduce FPS with `--fps` parameter (try --fps 2)
- Disable auto-fit and use manual zoom
- Close unused windows (puzzle browser, history)

**Episode history growing too large:**
- Episode recorder keeps max 500 episodes in memory
- Clear history with "CLEAR ALL" button in history viewer
- Save important episodes to file before clearing

**Replay not working:**
- Ensure episode has grid states recorded
- Grid states only recorded if episode completes
- Check that puzzle still exists in dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- New agent implementations (DQN, PPO, A2C, etc.)
- Additional visualization features
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## License

This project is created for the ARC Prize 2025 challenge.
Licensed under MIT License - see LICENSE file for details.

## Acknowledgments

- **ARC-AGI dataset** by François Chollet
- **ARC Prize 2025** challenge
- **Gymnasium** (OpenAI Gym successor)
- **Pygame** for visualization
- **Tkinter** for UI panels

## Citation

If you use this playground in your research, please cite:

```bibtex
@software{arc_agi_rl_playground,
  title = {ARC-AGI Reinforcement Learning Playground},
  author = {ARC-AGI RL Playground Contributors},
  year = {2025},
  url = {https://github.com/EmrullahAydogan/arc-agi-rl-playground}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub:
https://github.com/EmrullahAydogan/arc-agi-rl-playground/issues

---

**Happy Training! 🚀**
