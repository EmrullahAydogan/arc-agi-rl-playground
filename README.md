# ARC-AGI Reinforcement Learning Playground

Interactive training environment for reinforcement learning agents on ARC-AGI puzzles with real-time visualization and control.

## Features

### Dual-Window System
- **Two separate windows**: Main window (Pygame) for grids, Info Panel (Tkinter) for metrics
- **Multi-monitor support**: Drag the info panel to your second monitor
- **Independent windows**: Each window can be resized and positioned independently

### Main Window (Pygame)
- **Full-screen grid visualization**: INPUT, CURRENT, and TARGET grids side by side
- **Dynamic layouts**: Switch between horizontal (side-by-side) and vertical (stacked) layouts
- **Smart zoom system**:
  - Auto-fit mode: Automatically scales grids to fit window (8-40px range)
  - Manual zoom: Mouse wheel or Z/X keys
  - Adaptive: Automatically chooses best cell size for current grid
- **Resizable window**: Minimum 900x700, scales to any size
- **Button controls**: Clickable buttons for all major functions
- **ASCII-only display**: No unicode/emoji rendering issues

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

### Train/Test Mode System
- **Train Mode**: Learn from multiple training examples
  - Navigate between training samples with PREV/NEXT buttons
  - See input-output pairs for supervised learning
  - Track per-sample performance metrics
- **Test Mode**: Evaluate on the test puzzle
  - Agent predicts the output for the test input
  - Test-specific metrics tracking
- **Easy switching**: Toggle between modes with a button click

### Performance Tracking
- **Episode metrics**: Track success rate, accuracy, rewards
- **Per-sample tracking**: See which training samples are mastered
- **Real-time updates**: Metrics update as the agent trains
- **Historical data**: Keep track of all attempts and improvements

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/arc_prize_2025_RL.git
cd arc_prize_2025_RL
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

#### Button Controls (Recommended)
The main window features clickable buttons at the bottom:

| Button | Function |
|--------|----------|
| `> START` / `|| PAUSE` | Start/pause the agent |
| `RESET` | Reset current episode |
| `<> MODE` | Switch between Train/Test modes |
| `< PREV` / `NEXT >` | Navigate training samples |
| `SPEED -` / `SPEED +` | Adjust visualization speed |
| `LAYOUT` | Toggle horizontal/vertical layout |

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
| `Mouse Wheel` | Zoom in/out |

### Understanding the Display

**Main Window (Pygame):**
- **INPUT**: The starting grid for the current sample
- **CURRENT**: The agent's working grid (highlighted in yellow)
- **TARGET**: The goal grid to match

**Info Panel (Tkinter):**
- **STATUS**: Current episode information
- **AGENT PARAMETERS**: Agent hyperparameters
- **PERFORMANCE METRICS**: Training and testing statistics
- **SAMPLE STATUS**: Which training samples are solved

**Header Information:**
- Puzzle ID and dataset
- Current mode (TRAIN/TEST)
- Sample number (in train mode)
- Sample accuracy and solved status (color-coded)

### Training Workflow

1. **Start in Train Mode**: System starts paused in train mode
2. **Click START**: Agent begins attempting the first training sample
3. **Navigate Samples**: Use PREV/NEXT to try different training examples
4. **Monitor Progress**: Watch metrics in the info panel
5. **Switch to Test**: Click MODE button to evaluate on test puzzle
6. **Iterate**: Return to train mode to improve the agent

## Project Structure

```
arc_prize_2025_RL/
├── src/
│   ├── environment/
│   │   └── arc_env.py              # Gymnasium-compatible ARC environment
│   ├── visualization/
│   │   ├── pygame_viewer.py        # Main visualization window
│   │   └── info_panel.py           # Metrics/info panel (Tkinter)
│   ├── agents/
│   │   ├── base_agent.py           # Agent interface
│   │   └── random_agent.py         # Example random agent
│   └── utils/
│       ├── data_loader.py          # ARC data loader
│       └── metrics_tracker.py      # Performance tracking
├── arc-prize-2025/                 # ARC dataset files
├── playground.py                   # Main entry point
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
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

## Roadmap

- [ ] Add DQN agent implementation
- [ ] Add PPO agent implementation
- [ ] Multi-puzzle training
- [ ] Curriculum learning
- [ ] Action space extensions (transformations, patterns)
- [ ] Custom reward shaping
- [ ] Model checkpointing and loading
- [ ] Experiment tracking integration
- [ ] Multi-agent comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is created for the ARC Prize 2025 challenge.

## Acknowledgments

- ARC-AGI dataset by François Chollet
- ARC Prize 2025
- Gymnasium (OpenAI Gym successor)

## Screenshots

The playground features:
- Real-time grid visualization
- Live performance metrics
- Interactive controls
- Multi-monitor support
- Clean ASCII-only display (no rendering issues)

## Troubleshooting

**Grid dimensions mismatch error:**
- Fixed in latest version with adaptive grid slicing

**Unicode symbols not displaying:**
- All unicode replaced with ASCII for better compatibility

**Windows not appearing:**
- Check if pygame and tkinter are properly installed
- Verify display is available (not SSH without X11)

**Performance issues:**
- Reduce FPS with `--fps` parameter
- Disable auto-fit and use manual zoom

## Contact

For questions or suggestions, please open an issue on GitHub.
