#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "RLGameWrapperBase.cartridge_title": False,
    "RLGameWrapperBase.post_tick": False,
}

import abc
import io
import json
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

import pyboy
from pyboy.plugins.base_plugin import PyBoyGameWrapper
from pyboy.api.constants import TILES, SPRITES

logger = pyboy.logging.get_logger(__name__)


@dataclass
class RLState:
    """Data class representing a game state for RL"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
    action_mask: Optional[np.ndarray] = None
    state_id: Optional[int] = None


@dataclass
class RLAction:
    """Data class representing an action in the action space"""
    name: str
    button: str
    duration: int = 1
    value: int = 1


@dataclass
class EpisodeMetrics:
    """Data class for tracking episode metrics"""
    episode: int
    total_reward: float
    length: int
    start_time: float
    end_time: float
    final_score: float
    game_over: bool
    additional_metrics: Dict[str, Any]


class RLGameWrapperBase(PyBoyGameWrapper, abc.ABC):
    """
    Base class for Reinforcement Learning game wrappers.

    This class extends PyBoyGameWrapper with RL-specific functionality including:
    - Action space definition and mapping
    - Reward calculation
    - State representation and serialization
    - Episode tracking and metrics
    - Performance visualization
    """

    cartridge_title = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RL-specific attributes
        self.action_space = self._define_action_space()
        self.reward_weights = self._define_reward_weights()
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_start_time = None
        self.last_state = None
        self.state_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.metrics_history = []

        # Performance tracking
        self.performance_stats = {
            'avg_reward': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'fps': deque(maxlen=100)
        }

        # State serialization
        self.state_buffer = io.BytesIO()
        self.state_compression = kwargs.get('state_compression', True)

        # Visualization
        self.enable_visualization = kwargs.get('enable_visualization', False)
        self.viz_update_interval = kwargs.get('viz_update_interval', 10)

    @abc.abstractmethod
    def _define_action_space(self) -> List[RLAction]:
        """Define the action space for this game."""
        pass

    @abc.abstractmethod
    def _define_reward_weights(self) -> Dict[str, float]:
        """Define reward weights for different game events."""
        pass

    @abc.abstractmethod
    def _calculate_reward(self, prev_state: Optional[Dict], curr_state: Dict) -> float:
        """Calculate reward based on state changes."""
        pass

    @abc.abstractmethod
    def _get_game_state(self) -> Dict[str, Any]:
        """Extract relevant game state information."""
        pass

    @abc.abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get the observation array for the current state."""
        pass

    @abc.abstractmethod
    def _check_done(self) -> bool:
        """Check if the episode is done."""
        pass

    def reset(self, timer_div: Optional[int] = None) -> RLState:
        """
        Reset the game and return initial state.

        Args:
            timer_div: Optional timer divider value for randomization

        Returns:
            RLState: Initial state observation
        """
        if self.game_has_started:
            self.reset_game(timer_div=timer_div)
        else:
            self.start_game(timer_div=timer_div)

        # Reset episode tracking
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_start_time = time.time()
        self.state_history.clear()
        self.reward_history.clear()
        self.action_history.clear()

        # Get initial state
        game_state = self._get_game_state()
        observation = self._get_observation()
        self.last_state = game_state

        return RLState(
            observation=observation,
            reward=0.0,
            done=False,
            info=game_state,
            action_mask=self._get_action_mask()
        )

    def step(self, action: Union[int, str]) -> RLState:
        """
        Execute an action and return the new state.

        Args:
            action: Action to execute (index or name)

        Returns:
            RLState: New state after executing action
        """
        # Convert action to RLAction if needed
        if isinstance(action, int):
            if action < 0 or action >= len(self.action_space):
                raise ValueError(f"Invalid action index: {action}")
            rl_action = self.action_space[action]
        elif isinstance(action, str):
            rl_action = next((a for a in self.action_space if a.name == action), None)
            if rl_action is None:
                raise ValueError(f"Invalid action name: {action}")
        else:
            raise TypeError("Action must be int or str")

        # Execute action
        self._execute_action(rl_action)

        # Tick the emulator
        for _ in range(rl_action.duration):
            self.pyboy.tick(1, False)

        # Get new state
        current_state = self._get_game_state()
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(self.last_state, current_state)

        # Check if episode is done
        done = self._check_done()

        # Update tracking
        self.step_count += 1
        self.total_reward += reward
        self.state_history.append(current_state)
        self.reward_history.append(reward)
        self.action_history.append(rl_action.name)

        # Update performance stats
        self._update_performance_stats(reward, done)

        # Prepare info dict
        info = {
            **current_state,
            'step': self.step_count,
            'episode': self.episode_count,
            'total_reward': self.total_reward,
            'action_taken': rl_action.name
        }

        # Handle episode end
        if done:
            self._end_episode(current_state)

        self.last_state = current_state

        return RLState(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action_mask=self._get_action_mask(),
            state_id=self.step_count
        )

    def _execute_action(self, action: RLAction):
        """Execute a single action on the emulator."""
        if action.value > 0:
            self.pyboy.button_press(action.button)
        else:
            self.pyboy.button_release(action.button)

    def _get_action_mask(self) -> Optional[np.ndarray]:
        """
        Get action mask indicating which actions are valid.

        Returns:
            Optional[np.ndarray]: Boolean array of valid actions, or None if all actions are valid
        """
        # Default implementation allows all actions
        return None

    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(self.action_space)

    def get_action_names(self) -> List[str]:
        """Get list of action names."""
        return [action.name for action in self.action_space]

    def save_state(self) -> bytes:
        """
        Save current emulator state to bytes.

        Returns:
            bytes: Serialized state
        """
        self.state_buffer.seek(0)
        self.state_buffer.truncate()
        self.pyboy.save_state(self.state_buffer)
        return self.state_buffer.getvalue()

    def load_state(self, state_data: bytes):
        """
        Load emulator state from bytes.

        Args:
            state_data: Serialized state to load
        """
        self.state_buffer.seek(0)
        self.state_buffer.write(state_data)
        self.state_buffer.seek(0)
        self.pyboy.load_state(self.state_buffer)

    def serialize_state(self, state: RLState) -> bytes:
        """
        Serialize an RLState to bytes.

        Args:
            state: RLState to serialize

        Returns:
            bytes: Serialized state
        """
        state_dict = {
            'observation': state.observation.tobytes(),
            'observation_shape': state.observation.shape,
            'observation_dtype': str(state.observation.dtype),
            'reward': state.reward,
            'done': state.done,
            'info': state.info,
            'action_mask': state.action_mask.tobytes() if state.action_mask is not None else None,
            'state_id': state.state_id
        }

        return json.dumps(state_dict, default=str).encode('utf-8')

    def deserialize_state(self, data: bytes) -> RLState:
        """
        Deserialize bytes to RLState.

        Args:
            data: Serialized state data

        Returns:
            RLState: Deserialized state
        """
        state_dict = json.loads(data.decode('utf-8'))

        observation = np.frombuffer(
            state_dict['observation'],
            dtype=np.dtype(state_dict['observation_dtype'])
        ).reshape(state_dict['observation_shape'])

        action_mask = None
        if state_dict['action_mask'] is not None:
            action_mask = np.frombuffer(state_dict['action_mask'], dtype=np.bool_)

        return RLState(
            observation=observation,
            reward=state_dict['reward'],
            done=state_dict['done'],
            info=state_dict['info'],
            action_mask=action_mask,
            state_id=state_dict['state_id']
        )

    def _update_performance_stats(self, reward: float, done: bool):
        """Update performance statistics."""
        self.performance_stats['avg_reward'].append(reward)

        if done:
            self.performance_stats['episode_lengths'].append(self.step_count)
            success = self.step_count > 100  # Arbitrary success threshold
            self.performance_stats['success_rate'].append(float(success))

    def _end_episode(self, final_state: Dict[str, Any]):
        """Handle episode end and record metrics."""
        episode_time = time.time() - self.episode_start_time

        metrics = EpisodeMetrics(
            episode=self.episode_count,
            total_reward=self.total_reward,
            length=self.step_count,
            start_time=self.episode_start_time,
            end_time=time.time(),
            final_score=final_state.get('score', 0),
            game_over=True,
            additional_metrics={
                'avg_fps': self.step_count / episode_time if episode_time > 0 else 0,
                **final_state
            }
        )

        self.metrics_history.append(metrics)
        logger.info(f"Episode {self.episode_count} completed - "
                   f"Reward: {self.total_reward:.2f}, "
                   f"Length: {self.step_count}, "
                   f"Time: {episode_time:.2f}s")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.

        Returns:
            Dict[str, Any]: Performance summary
        """
        if not self.metrics_history:
            return {}

        recent_episodes = self.metrics_history[-10:]  # Last 10 episodes

        summary = {
            'total_episodes': len(self.metrics_history),
            'recent_avg_reward': np.mean([m.total_reward for m in recent_episodes]),
            'recent_avg_length': np.mean([m.length for m in recent_episodes]),
            'recent_success_rate': np.mean([1 if m.length > 100 else 0 for m in recent_episodes]),
            'best_reward': max(m.total_reward for m in self.metrics_history),
            'avg_fps': np.mean([m.additional_metrics.get('avg_fps', 0) for m in recent_episodes])
        }

        return summary

    def export_metrics(self, filepath: str):
        """
        Export metrics history to JSON file.

        Args:
            filepath: Path to save metrics
        """
        export_data = {
            'episodes': [asdict(metrics) for metrics in self.metrics_history],
            'performance_summary': self.get_performance_summary(),
            'action_space': [asdict(action) for action in self.action_space],
            'reward_weights': self.reward_weights
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")

    def __repr__(self):
        base_repr = super().__repr__()
        performance = self.get_performance_summary()

        if performance:
            rl_info = (
                f"\nRL Stats:\n"
                f"  Episodes: {performance['total_episodes']}\n"
                f"  Recent Avg Reward: {performance['recent_avg_reward']:.2f}\n"
                f"  Recent Success Rate: {performance['recent_success_rate']:.2%}\n"
                f"  Best Reward: {performance['best_reward']:.2f}\n"
                f"  Action Space Size: {len(self.action_space)}\n"
            )
            return base_repr + rl_info

        return base_repr + f"\nRL Stats: No episodes completed yet"