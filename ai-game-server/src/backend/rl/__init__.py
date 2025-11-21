"""
PyBoy Reinforcement Learning Environment Module

This module provides a comprehensive RL environment for Game Boy games using PyBoy emulator.

Features:
- Gym-compatible environment interface
- Memory-based reward systems
- Multiple observation spaces (screen, game area, memory, tiles)
- Various action space types (discrete, multi-discrete, continuous, hybrid)
- Enhanced game state tracking
- Training pipeline integration with Stable Baselines3
- Visualization and monitoring tools
- Game-specific configurations

Quick Start:
    ```python
    from ai_game_server.src.backend.rl import PyBoyEnv, RLEnvironmentConfig

    # Create environment
    config = RLEnvironmentConfig(
        observation_config=ObservationConfig(type="screen"),
        action_config=ActionConfig(type="discrete")
    )

    env = PyBoyEnv("game.gb", config=config)

    # Run environment
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    ```

Training:
    ```python
    from ai_game_server.src.backend.rl.training_pipeline import create_training_pipeline

    # Create training pipeline
    pipeline = create_training_pipeline(
        rom_path="game.gb",
        config=TrainingConfig(algorithm="PPO", total_timesteps=100000)
    )

    # Train model
    pipeline.create_model()
    pipeline.train()
    ```

Classes:
    PyBoyEnv: Main RL environment class
    RLEnvironmentConfig: Environment configuration
    TrainingPipeline: Training pipeline with SB3 integration
    MemoryRewardSystem: Memory-based reward system
    GameStateTracker: Enhanced state tracking
    ActionManager: Action space management
    LiveVisualizer: Real-time visualization
"""

# Core classes
from .pyboy_env import PyBoyEnv, RLStepResult
from .rl_environment_config import (
    RLEnvironmentConfig,
    ObservationConfig,
    ActionConfig,
    RewardConfig,
    StateTrackingConfig,
    ObservationType,
    ActionType
)

# Reward system
from .memory_reward_system import (
    MemoryRewardSystem,
    RewardSystemConfig,
    RewardType,
    MemoryAddress,
    RewardConfig
)

# State tracking
from .game_state_tracker import (
    GameStateTracker,
    StateTrackingConfig,
    GameStateType,
    GameStateSnapshot,
    ScreenAnalysis,
    SpriteInfo
)

# Action management
from .action_manager import (
    ActionManager,
    ActionConfig,
    ActionType,
    ButtonAction,
    ActionMapping
)

# Training pipeline
from .training_pipeline import (
    TrainingPipeline,
    TrainingConfig,
    EvaluationConfig,
    TrainingMetrics,
    CustomCallback,
    create_training_pipeline,
    quick_train
)

# Visualization
from .visualization import (
    LiveVisualizer,
    VideoRecorder,
    VisualizationConfig
)

# Version info
__version__ = "1.0.0"
__author__ = "PyBoy RL Team"
__email__ = "contact@pyboy-rl.org"
__license__ = "MIT"

# Make sure PyBoy is available
try:
    import pyboy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False

# Check for optional dependencies
try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import stable_baselines3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Export main classes for easier import
__all__ = [
    # Core
    'PyBoyEnv',
    'RLStepResult',
    'RLEnvironmentConfig',

    # Configuration
    'ObservationConfig',
    'ActionConfig',
    'RewardConfig',
    'StateTrackingConfig',
    'ObservationType',
    'ActionType',

    # Reward system
    'MemoryRewardSystem',
    'RewardSystemConfig',
    'RewardType',
    'MemoryAddress',

    # State tracking
    'GameStateTracker',
    'GameStateType',
    'GameStateSnapshot',
    'ScreenAnalysis',
    'SpriteInfo',

    # Action management
    'ActionManager',
    'ButtonAction',
    'ActionMapping',

    # Training
    'TrainingPipeline',
    'TrainingConfig',
    'EvaluationConfig',
    'TrainingMetrics',
    'create_training_pipeline',
    'quick_train',

    # Visualization
    'LiveVisualizer',
    'VideoRecorder',
    'VisualizationConfig',

    # Version flags
    'PYBOY_AVAILABLE',
    'GYM_AVAILABLE',
    'SB3_AVAILABLE',
    'TORCH_AVAILABLE',
    '__version__',
]

def check_dependencies():
    """Check if all required dependencies are available."""
    issues = []

    if not PYBOY_AVAILABLE:
        issues.append("PyBoy is not available. Install with 'pip install pyboy'")

    if not GYM_AVAILABLE:
        issues.append("OpenAI Gym is not available. Install with 'pip install gym'")

    if not SB3_AVAILABLE:
        issues.append("Stable Baselines3 is not available. Install with 'pip install stable-baselines3'")

    if issues:
        print("Dependency issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nInstall missing dependencies to use all features.")
        return False

    print("All dependencies are available!")
    return True

def get_game_config(game_name: str) -> RLEnvironmentConfig:
    """Get pre-configured environment for specific games."""
    config = RLEnvironmentConfig()
    config.update_for_game(game_name)
    return config

def create_env_from_rom(rom_path: str, **kwargs) -> PyBoyEnv:
    """Create PyBoyEnv from ROM path with automatic game detection."""
    config = RLEnvironmentConfig(**kwargs)

    # Try to detect game from ROM filename
    rom_name = rom_path.lower()

    if "pokemon" in rom_name:
        config.update_for_game("pokemon")
    elif "mario" in rom_name:
        config.update_for_game("mario")
    elif "tetris" in rom_name:
        config.update_for_game("tetris")
    elif "zelda" in rom_name:
        config.update_for_game("zelda")

    return PyBoyEnv(rom_path, config)

# Print welcome message
def print_welcome():
    """Print welcome message and usage information."""
    print("=" * 60)
    print("PyBoy Reinforcement Learning Environment")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()

    print("Quick Start:")
    print("  from ai_game_server.src.backend.rl import PyBoyEnv")
    print("  env = PyBoyEnv('game.gb')")
    print("  obs = env.reset()")
    print("  action = env.action_space.sample()")
    print("  obs, reward, done, info = env.step(action)")
    print()

    print("Training:")
    print("  from ai_game_server.src.backend.rl.training_pipeline import create_training_pipeline")
    print("  pipeline = create_training_pipeline('game.gb')")
    print("  pipeline.create_model()")
    print("  pipeline.train()")
    print()

    print("For more information, check the documentation.")
    print("=" * 60)

# Print welcome message when module is imported
if __name__ != "__main__":
    print_welcome()