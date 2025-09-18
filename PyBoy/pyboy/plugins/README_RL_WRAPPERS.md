# PyBoy RL Game Wrappers

Enhanced Reinforcement Learning game wrappers for PyBoy with advanced RL capabilities, performance tracking, and visualization tools.

## Overview

This enhancement adds comprehensive RL functionality to existing PyBoy game wrappers, enabling sophisticated reinforcement learning experiments with Game Boy games. The wrappers maintain backward compatibility while adding powerful RL-specific features.

## Features

### 🎮 Enhanced Game Wrappers

- **Super Mario Land RL**: Platformer-specific RL with progress tracking, enemy detection, and skill assessment
- **Tetris RL**: Puzzle game RL with board heuristics, piece placement optimization, and line-clearing strategies
- **Pokemon Gen 1 RL**: RPG-style RL with exploration, battle mechanics, and completionist tracking
- **General Game Boy RL**: Universal wrapper for any Game Boy game using computer vision and adaptive learning

### 🧠 RL Capabilities

- **Action Spaces**: Game-specific action definitions with proper timing and combinations
- **Reward Systems**: Sophisticated reward calculation with game-specific heuristics
- **State Observations**: Multiple observation types (tiles, pixels, hybrid, analysis-based)
- **Episode Management**: Complete episode lifecycle with reset/step functionality
- **State Serialization**: Save/load game states for reproducible experiments

### 📊 Performance Tracking

- **Real-time Metrics**: Episode rewards, lengths, success rates, and learning progress
- **Game-specific Analytics**: Custom metrics for each game type (lines cleared, Pokemon caught, etc.)
- **Skill Assessment**: Automatic evaluation of agent skill level and improvement
- **Performance Reports**: Comprehensive reports with recommendations

### 📈 Visualization & Analysis

- **Live Dashboards**: Real-time training visualization with matplotlib
- **Interactive Plots**: Plotly-based interactive dashboards
- **Performance Reports**: Detailed analysis and recommendations
- **Export Capabilities**: JSON metrics export and image generation

## Installation

### Prerequisites

```bash
# Install PyBoy (if not already installed)
pip install pyboy

# Install optional dependencies for visualization
pip install matplotlib seaborn plotly opencv-python
```

### Setup

1. **Copy RL wrapper files** to `PyBoy/pyboy/plugins/`:
   - `rl_game_wrapper_base.py`
   - `rl_game_wrapper_super_mario_land.py`
   - `rl_game_wrapper_tetris.py`
   - `rl_game_wrapper_pokemon_gen1.py`
   - `rl_game_wrapper_general.py`
   - `rl_visualization.py`
   - `rl_examples.py`
   - `manager_rl_gen.py`

2. **Update plugin manager**:
   ```bash
   cd PyBoy/pyboy/plugins/
   python manager_rl_gen.py
   ```

3. **Build PyBoy**:
   ```bash
   cd PyBoy
   make build_pyboy
   ```

## Quick Start

### Basic Usage

```python
import pyboy
from pyboy import PyBoy

# Initialize PyBoy with RL wrapper
pyboy = PyBoy(
    "path/to/game.gb",
    window_type="null",  # Headless for training
    game_wrapper=True    # Enable game wrapper
)

# Get RL wrapper
wrapper = pyboy.game_wrapper

# Reset environment
state = wrapper.reset()

# Take an action
next_state = wrapper.step(0)  # Action index

# Access game-specific information
print(f"Reward: {next_state.reward}")
print(f"Done: {next_state.done}")
print(f"Info: {next_state.info}")
```

### Advanced Usage with Visualization

```python
from pyboy.plugins.rl_visualization import RLVisualizer, TrainingMetrics

# Create visualizer
viz = RLVisualizer("training_output")

# Training loop
for episode in range(100):
    state = wrapper.reset()
    total_reward = 0

    while not state.done:
        action = agent.get_action(state.observation)  # Your RL agent
        next_state = wrapper.step(action)
        total_reward += next_state.reward

        # Update your agent here...
        state = next_state

    # Record metrics
    metrics = TrainingMetrics(
        episode=episode,
        step=state.info.get('step', 0),
        reward=total_reward,
        length=state.info.get('step', 0),
        survival_time=state.info.get('step', 0) / 60.0,
        final_score=wrapper.score,
        success=total_reward > 100,
        game_specific_metrics=wrapper.get_game_state(),
        timestamp=time.time()
    )

    viz.add_metrics(metrics)

# Generate final report
print(viz.generate_performance_report())
```

## Game-Specific Documentation

### Super Mario Land RL

```python
# Initialize with specific settings
pyboy = PyBoy("super_mario_land.gb", game_wrapper=True)
wrapper = pyboy.game_wrapper

# Access Mario-specific features
print(f"World: {wrapper.world}")
print(f"Level Progress: {wrapper.level_progress}")
print(f"Enemies Defeated: {wrapper.enemies_defeated}")

# Get skill assessment
skill = wrapper.get_learning_progression()
print(f"Skill Level: {skill['skill_progression']['position_trend']}")
```

**Action Space:**
- `no_op`, `jump`, `run`, `jump_run`, `left`, `right`, `jump_left`, `jump_right`, `run_left`, `run_right`, `run_jump_left`, `run_jump_right`

**Reward Components:**
- Progress奖励 (rightward movement)
- Coin collection
- Enemy defeat
- Power-up collection
- Level completion
- Survival bonus

### Tetris RL

```python
# Initialize with RL-specific options
pyboy = PyBoy("tetris.gb", game_wrapper=True)
wrapper = pyboy.game_wrapper

# Access Tetris-specific analytics
stats = wrapper.get_tetris_statistics()
print(f"Lines per piece: {stats['lines_per_piece']}")
print(f"Tetris rate: {stats['tetris_rate']}")

# Get skill assessment
skill = wrapper.get_skill_assessment()
print(f"Skill Level: {skill['skill_level']}")
```

**Action Space:**
- `left`, `right`, `down`, `rotate_left`, `rotate_right`, `hard_drop`
- `fast_left`, `fast_right`, `soft_drop`, `hold_piece`, `wait`

**Reward Components:**
- Line clearing (singles, doubles, triples, tetrises)
- Combo bonuses
- Perfect clears
- Board heuristics (holes, height, bumpiness)
- Survival time

### Pokemon Gen 1 RL

```python
# Initialize for Pokemon Red/Blue
pyboy = PyBoy("pokemon_red.gb", game_wrapper=True)
wrapper = pyboy.game_wrapper

# Track progress
pokedex = wrapper.get_pokedex_progress()
gym = wrapper.get_gym_progress()
exploration = wrapper.get_exploration_progress()

print(f"Pokedex: {pokedex['pokemon_caught']}/151")
print(f"Badges: {gym['badges_earned']}/8")
print(f"Areas explored: {exploration['unique_maps_count']}")
```

**Action Space:**
- Movement: `up`, `down`, `left`, `right`
- Actions: `a_button`, `b_button`, `start`, `select`
- Combinations: `up_left`, `up_right`, `down_left`, `down_right`
- Menu actions: `talk_interact`, `menu_cancel`

**Reward Components:**
- Experience gain
- Pokemon caught
- Badge earned
- Money gained
- Trainer/wild Pokemon defeated
- New area exploration

### General Game Boy RL

```python
# Works with any Game Boy game
pyboy = PyBoy("any_game.gb", game_wrapper=True)
wrapper = pyboy.game_wrapper

# Automatic game analysis
analysis = wrapper.get_game_analysis()
print(f"Game Type: {analysis['game_type']}")
print(f"Difficulty: {analysis['estimated_difficulty']}")

# Adaptive rewards based on performance
wrapper.adapt_rewards()
```

**Features:**
- Automatic game type detection
- Computer vision-based state observation
- Adaptive reward systems
- Score and progress detection
- Universal action space

## Visualization Tools

### Real-time Dashboard

```python
from pyboy.plugins.rl_visualization import create_visualization_suite

# Create complete visualization suite
viz_suite = create_visualization_suite("my_training")

# Training loop with monitoring
for episode in range(num_episodes):
    # ... training code ...

    # Add metrics to visualizer
    viz_suite['visualizer'].add_metrics(metrics)

    # Real-time monitoring
    stats = viz_suite['monitor'].get_current_stats()
    print(f"Episodes per minute: {stats['eps_per_minute']}")
```

### Performance Reports

```python
# Generate comprehensive report
report = viz.generate_performance_report()
print(report)

# Export metrics
viz.export_metrics("training_data.json")

# Create interactive dashboard
if hasattr(viz, 'create_interactive_dashboard'):
    dashboard_path = viz.create_interactive_dashboard()
```

## Configuration Options

### Wrapper Configuration

```python
# Super Mario Land specific
wrapper_config = {
    'observation_type': 'hybrid',      # 'tiles', 'pixels', 'hybrid'
    'frame_stack_size': 4,
    'aggressive_mode': False,
    'survival_mode': False
}

# Tetris specific
wrapper_config = {
    'observation_type': 'analysis',
    'include_next_piece': True,
    'include_heuristics': True,
    'aggressive_mode': True
}

# General configuration
wrapper_config = {
    'vision_resolution': (84, 84),
    'use_grayscale': True,
    'use_frame_diff': True,
    'adaptive_rewards': True
}
```

### Visualization Configuration

```python
viz_config = {
    'output_dir': 'training_output',
    'auto_save': True,
    'update_interval': 10,  # Update every 10 episodes
    'plot_styles': {
        'reward': 'blue',
        'length': 'green',
        'success': 'orange'
    }
}
```

## Examples

Run the comprehensive examples:

```bash
cd PyBoy/pyboy/plugins/
python rl_examples.py
```

Available examples:
1. Super Mario Land RL
2. Tetris RL
3. Pokemon Gen 1 RL
4. General Game Boy RL
5. Visualization Demo

## Performance Tips

### Optimization

1. **Headless Mode**: Use `window_type="null"` for training
2. **Frame Skipping**: Use `pyboy.tick(15, False)` to skip frames
3. **Observation Types**: Choose appropriate observation complexity
4. **Action Duration**: Optimize action timing for each game

### Memory Management

1. **State Serialization**: Use for reproducible experiments
2. **Frame Stacking**: Manage memory usage with appropriate stack sizes
3. **Metrics Buffer**: Limit metrics history to prevent memory issues

### Training Strategies

1. **Curriculum Learning**: Start with simpler tasks
2. **Reward Shaping**: Balance immediate and delayed rewards
3. **Exploration**: Use appropriate epsilon decay rates
4. **Evaluation**: Regular performance assessment

## API Reference

### RLGameWrapperBase

Base class for all RL wrappers.

#### Methods

- `reset(timer_div=None)`: Reset environment and return initial state
- `step(action)`: Execute action and return new state
- `get_action_space_size()`: Get number of available actions
- `get_action_names()`: Get list of action names
- `save_state()`: Save current emulator state
- `load_state(state_data)`: Load emulator state
- `get_performance_summary()`: Get performance statistics

#### Returns

`RLState` object with:
- `observation`: Current state observation
- `reward`: Reward for last action
- `done`: Whether episode is finished
- `info`: Additional game-specific information
- `action_mask`: Valid actions mask
- `state_id`: Unique state identifier

### Visualization Classes

#### RLVisualizer

- `add_metrics(metrics)`: Add training metrics
- `update_plots()`: Update all plots
- `save_current_plots(filename)`: Save plots to file
- `generate_performance_report()`: Generate text report
- `export_metrics(filename)`: Export metrics to JSON

#### TrainingMetrics

Data class for tracking episode metrics:
- `episode`, `step`, `reward`, `length`, `survival_time`
- `final_score`, `success`, `game_specific_metrics`
- `timestamp`, `learning_rate`, `epsilon`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **ROM Not Found**: Provide correct ROM file paths
3. **Performance Issues**: Use headless mode and optimize observations
4. **Memory Issues**: Limit metrics history and frame stack size

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check wrapper type
print(type(pyboy.game_wrapper))

# Test basic functionality
state = wrapper.reset()
print(f"Action space size: {wrapper.get_action_space_size()}")
print(f"Observation shape: {state.observation.shape}")
```

## Contributing

### Adding New Games

1. Create new wrapper class inheriting from `RLGameWrapperBase`
2. Implement required abstract methods:
   - `_define_action_space()`
   - `_define_reward_weights()`
   - `_calculate_reward()`
   - `_get_game_state()`
   - `_get_observation()`
   - `_check_done()`
3. Add game-specific features and metrics
4. Update plugin manager configuration

### Extending Visualization

1. Add new plot types to `RLVisualizer`
2. Implement game-specific metrics
3. Create custom report generators
4. Add export formats

## License

This project follows the same license as PyBoy. See LICENSE.md for details.

## Acknowledgments

- PyBoy development team for the original emulator
- OpenAI Gym for RL environment design patterns
- Stable Baselines for RL implementation inspiration
- Matplotlib, Seaborn, and Plotly for visualization tools

---

For more information, examples, and updates, please refer to the [PyBoy documentation](https://baekalfen.github.io/PyBoy/) and the [examples](rl_examples.py) provided.