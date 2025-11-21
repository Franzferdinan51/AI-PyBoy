# PyBoy RL Implementation Documentation

This document provides a comprehensive overview of the PyBoyEnv reinforcement learning implementation integrated into the existing PyBoy system.

## Overview

The PyBoy RL implementation provides a complete reinforcement learning environment for Game Boy games using the PyBoy emulator. It features:

- **Gym-compatible interface**: Full OpenAI Gym compatibility
- **Memory-based reward systems**: Advanced reward calculation using game memory
- **Multiple observation spaces**: Screen, game area, memory, and multi-modal
- **Flexible action spaces**: Discrete, multi-discrete, continuous, and hybrid
- **Enhanced state tracking**: Comprehensive game state monitoring
- **Training pipeline integration**: Stable Baselines3 support
- **Visualization tools**: Real-time monitoring and analysis
- **Game-specific configurations**: Optimized settings for popular games

## Architecture

### Core Components

```
PyBoyEnv (Main Environment)
├── MemoryRewardSystem
├── GameStateTracker
├── ActionManager
├── RLEnvironmentConfig
└── PyBoy (Emulator)
```

### Module Structure

- **`pyboy_env.py`**: Main RL environment class with gym interface
- **`memory_reward_system.py`**: Memory-based reward calculation
- **`game_state_tracker.py`**: Enhanced game state monitoring
- **`action_manager.py`**: Action space management
- **`rl_environment_config.py`**: Configuration management
- **`training_pipeline.py`**: Stable Baselines3 integration
- **`visualization.py`**: Real-time visualization tools

## Installation

### Dependencies

```bash
# Core dependencies
pip install pyboy gym numpy

# Optional dependencies for training
pip install stable-baselines3 torch

# Optional dependencies for visualization
pip install matplotlib seaborn plotly opencv-python Pillow

# Development dependencies
pip install pytest pytest-benchmark
```

### Verification

```python
from src.backend.rl import check_dependencies
check_dependencies()
```

## Quick Start

### Basic Usage

```python
from src.backend.rl import PyBoyEnv, RLEnvironmentConfig

# Create environment
env = PyBoyEnv("game.gb")

# Run environment
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

### Training with Stable Baselines3

```python
from src.backend.rl.training_pipeline import create_training_pipeline

# Create training pipeline
pipeline = create_training_pipeline(
    rom_path="game.gb",
    config=TrainingConfig(algorithm="PPO", total_timesteps=100000)
)

# Train model
pipeline.create_model()
pipeline.train()
```

## Configuration

### Environment Configuration

```python
from src.backend.rl import RLEnvironmentConfig, ObservationConfig, ActionConfig

config = RLEnvironmentConfig(
    # Basic settings
    headless=True,
    frames_per_action=4,
    max_steps=1000,

    # Observation configuration
    observation_config=ObservationConfig(
        type="screen",  # screen, game_area, memory, tiles, multi
        grayscale=False,
        stack_frames=1
    ),

    # Action configuration
    action_config=ActionConfig(
        type="discrete",  # discrete, multi_discrete, continuous, hybrid
        include_noop=True,
        available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
    ),

    # Reward system
    reward_configs=[
        {
            "type": "exploration",
            "weight": 0.1,
            "memory_addresses": [
                {"address": 0xC0A0, "name": "position_x"},
                {"address": 0xC0A1, "name": "position_y"}
            ]
        }
    ]
)

env = PyBoyEnv("game.gb", config=config)
```

### Game-Specific Configurations

The system includes pre-configured settings for popular games:

```python
from src.backend.rl import get_game_config

# Get game-specific configuration
config = get_game_config("pokemon")  # pokemon, mario, tetris, zelda
env = PyBoyEnv("pokemon.gb", config=config)
```

## Observation Spaces

### Screen Observation
- **Format**: RGB array (144x160x3)
- **Features**: Full game screen capture
- **Preprocessing**: Optional grayscale, resizing, frame stacking

### Game Area Observation
- **Format**: Tile matrix (variable size)
- **Features**: Simplified game area using tiles and sprites
- **Optimization**: Reduced dimensionality for faster training

### Memory Observation
- **Format**: Byte array (customizable size)
- **Features**: Direct memory region access
- **Use Case**: Memory-based agents and analysis

### Multi-Modal Observation
- **Format**: Dictionary containing multiple observation types
- **Features**: Combined screen, game area, and memory data
- **Flexibility**: Customizable combination of observation types

## Action Spaces

### Discrete Actions
```python
# Single action from predefined set
action_space = Discrete(9)  # NOOP + 8 directions/buttons
```

### Multi-Discrete Actions
```python
# Multiple buttons can be pressed simultaneously
action_space = MultiDiscrete([2, 2, 2, 2])  # 4 binary buttons
```

### Continuous Actions
```python
# Continuous joystick-like input
action_space = Box(low=-1.0, high=1.0, shape=(2,))  # x, y direction
```

### Hybrid Actions
```python
# Combination of discrete and continuous
action_space = Dict({
    'discrete': MultiDiscrete([2, 2]),  # A, B buttons
    'continuous': Box(low=-1.0, high=1.0, shape=(2,))  # direction
})
```

## Reward System

### Memory-Based Rewards

The system monitors game memory to calculate rewards:

```python
# Example Pokemon rewards
reward_configs = [
    {
        "type": "experience",
        "memory_addresses": [{"address": 0xD18C, "name": "current_exp"}],
        "reward_on_increase": True,
        "weight": 1.0
    },
    {
        "type": "level",
        "memory_addresses": [{"address": 0xD18D, "name": "pokemon_level"}],
        "reward_on_increase": True,
        "weight": 5.0
    }
]
```

### Custom Reward Functions

```python
def exploration_reward(values, previous_values):
    """Calculate exploration reward based on position changes."""
    if len(values) >= 2 and len(previous_values) >= 2:
        dx = values[0] - previous_values[0]
        dy = values[1] - previous_values[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return min(distance * 0.1, 1.0)
    return 0.0

reward_configs = [
    {
        "type": "custom",
        "custom_function": exploration_reward,
        "weight": 0.1
    }
]
```

## State Tracking

### Enhanced Game State

The system tracks comprehensive game state:

```python
# Access game state
state = env.game_state_tracker.get_state()

# State includes:
# - Basic game information
# - Screen analysis (brightness, contrast, entropy)
# - Sprite information and positions
# - Memory region activity
# - Performance metrics
# - Custom tracking data
```

### Performance Monitoring

```python
# Get performance summary
perf_summary = env.game_state_tracker.get_performance_summary()

# Includes:
# - FPS (frames per second)
# - Tick and emulation times
# - Memory usage patterns
# - Screen motion analysis
```

## Training

### Supported Algorithms

- **PPO**: Proximal Policy Optimization
- **A2C**: Advantage Actor-Critic
- **DQN**: Deep Q-Network
- **SAC**: Soft Actor-Critic (for continuous actions)
- **TD3**: Twin Delayed DDPG (for continuous actions)

### Training Pipeline

```python
from src.backend.rl.training_pipeline import TrainingPipeline, TrainingConfig

# Configure training
training_config = TrainingConfig(
    algorithm="PPO",
    total_timesteps=1000000,
    learning_rate=3e-4,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./logs"
)

# Create and run training
pipeline = TrainingPipeline(env, training_config)
pipeline.create_model()
pipeline.train()

# Evaluate results
results = pipeline.evaluate()
```

### Hyperparameter Tuning

```python
# Quick training with different parameters
results = quick_train(
    rom_path="game.gb",
    algorithm="PPO",
    total_timesteps=50000,
    output_dir="./tuning_results"
)
```

## Visualization

### Live Visualization

```python
from src.backend.rl import LiveVisualizer, VisualizationConfig

# Configure visualization
viz_config = VisualizationConfig(
    window_size=(1200, 800),
    update_interval=0.05,
    save_screenshots=True,
    enable_live_plotting=True
)

# Create and start visualizer
visualizer = LiveVisualizer(env, viz_config)
visualizer.start()

# Run environment with real-time visualization
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    visualizer.update()
    if done:
        break

visualizer.stop()
```

### Video Recording

```python
from src.backend.rl import VideoRecorder

# Record gameplay video
with VideoRecorder(env, "gameplay.mp4", fps=30) as recorder:
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        recorder.record_frame()
        if done:
            break
```

### Analysis and Export

```python
# Create visualizations
visualizer.create_reward_animation("rewards.gif")
visualizer.create_memory_heatmap("memory_heatmap.png")
visualizer.create_interactive_dashboard("dashboard.html")

# Export data
visualizer.export_visualization_data("analysis.json")
```

## Game-Specific Implementations

### Pokemon Games

**Memory Addresses:**
- Experience: 0xD18C
- Level: 0xD18D
- Badges: 0xD362
- HP: 0xD16C

**Recommended Actions:**
- UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

**Observation:**
- Multi-modal (screen + memory)

### Super Mario Land

**Memory Addresses:**
- Score: 0xC0A0-0xC0A2 (16-bit)
- Coins: 0xC0AD
- Lives: 0xC07A

**Recommended Actions:**
- UP, DOWN, LEFT, RIGHT, A, B

**Observation:**
- Screen or game area

### Tetris

**Memory Addresses:**
- Score: 0xC0A0-0xC0A2 (16-bit)
- Lines: 0xC0B0
- Level: 0xC0AE

**Recommended Actions:**
- LEFT, RIGHT, DOWN, A (rotate), B (hard drop)

**Observation:**
- Game area (tiles)

## Performance Optimization

### Memory Usage

- **Headless mode**: Disable GUI for faster training
- **Frame skipping**: Process multiple frames per action
- **Simplified observations**: Use game_area instead of screen
- **Batched training**: Use larger batch sizes

### Training Speed

- **Vectorized environments**: Use multiple environments
- **GPU acceleration**: Enable CUDA support
- **Async processing**: Use background threads
- **Caching**: Cache repeated calculations

## Advanced Features

### Custom Tracking

```python
def custom_tracker(pyboy):
    """Custom game state tracking function."""
    return {
        "custom_metric": pyboy.memory[0xC000],
        "calculated_value": some_calculation(pyboy)
    }

# Add to environment
env.game_state_tracker.add_custom_tracker("my_tracker", custom_tracker)
```

### State Serialization

```python
# Save complete environment state
state_data = env.save_state()

# Load state later
env.load_state(state_data)
```

### Action Mappings

```python
# Define custom action mappings
from src.backend.rl.action_manager import ActionMapping

custom_mapping = ActionMapping(
    action_id=10,
    buttons=["UP", "A"],  # Jump action
    duration=4,
    description="Jump"
)

env.action_manager.add_custom_mapping(custom_mapping)
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install pyboy gym numpy stable-baselines3
```

**ROM Loading:**
- Ensure ROM file is valid Game Boy ROM
- Check file permissions
- Verify ROM path is correct

**Performance Issues:**
- Use headless mode for training
- Reduce observation complexity
- Increase frames_per_action
- Optimize reward calculations

**Memory Access Errors:**
- Verify memory addresses are correct for specific game
- Check if game uses memory bank switching
- Use game-specific configurations

### Debug Mode

Enable debug logging:
```python
config = RLEnvironmentConfig(
    log_level="DEBUG",
    track_performance=True
)
```

## API Reference

### PyBoyEnv

**Methods:**
- `reset()`: Reset environment to initial state
- `step(action)`: Execute action and return results
- `render(mode)`: Render environment
- `close()`: Clean up resources
- `save_state()`: Save environment state
- `load_state(state)`: Load environment state

**Properties:**
- `action_space`: Environment action space
- `observation_space`: Environment observation space
- `reward_range`: Range of possible rewards
- `metadata`: Environment metadata

### MemoryRewardSystem

**Methods:**
- `get_reward()`: Calculate current reward
- `get_reward_breakdown()`: Get reward component breakdown
- `is_done()`: Check if episode should end
- `reset()`: Reset reward tracking

### GameStateTracker

**Methods:**
- `update()`: Update game state tracking
- `get_state()`: Get current game state
- `get_memory_changes()`: Analyze memory changes
- `get_sprite_analysis()`: Analyze sprite positions
- `export_state(filepath)`: Export state data

## Examples

See the `examples/` directory for complete usage examples:

- `rl_basic_example.py`: Basic environment usage
- `rl_training_example.py`: Training with Stable Baselines3
- `rl_visualization_example.py`: Visualization and monitoring

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd pyboy-rl

# Install development dependencies
pip install -e .
pip install pytest pytest-benchmark

# Run tests
pytest tests/

# Run benchmarks
pytest tests/test_benchmark.py
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all public methods
- Include examples in docstrings

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [OpenAI Gym](https://gym.openai.com/) - RL environment interface

## Support

For issues and questions:
- GitHub Issues: [Project Issues]
- Documentation: [Project Wiki]
- Discussions: [Project Discussions]

---

*This implementation provides a comprehensive foundation for reinforcement learning research on Game Boy games.*