#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "RLGameWrapperSuperMarioLand.cartridge_title": False,
    "RLGameWrapperSuperMarioLand.post_tick": False,
}

import numpy as np
from typing import Dict, List, Optional, Any

import pyboy
from pyboy.utils import dec_to_bcd, bcd_to_dec
from pyboy.api.constants import TILES

from .game_wrapper_super_mario_land import GameWrapperSuperMarioLand, mapping_minimal, mapping_compressed
from .rl_game_wrapper_base import RLGameWrapperBase, RLAction, RLState

logger = pyboy.logging.get_logger(__name__)

# Memory addresses for Super Mario Land
ADDR_TIME_LEFT = 0xDA01
ADDR_LIVES_LEFT = 0xDA15
ADDR_LIVES_LEFT_DISPLAY = 0x9806
ADDR_WORLD_LEVEL = 0xFFB4
ADDR_WIN_COUNT = 0xFF9A
ADDR_MARIO_X = 0xC202
ADDR_MARIO_Y = 0xC203
ADDR_MARIO_STATE = 0xC0A4
ADDR_MARIO_VELOCITY_X = 0xC20A
ADDR_MARIO_VELOCITY_Y = 0xC20B
ADDR_SCORE = 0xDA06
ADDR_COINS = 0xDA0B

# Mario states
MARIO_STATE_NORMAL = 0x00
MARIO_STATE_DEAD = 0x39
MARIO_STATE_STAR_POWER = 0x0C
MARIO_STATE_INVINCIBLE = 0x08

class RLGameWrapperSuperMarioLand(RLGameWrapperBase, GameWrapperSuperMarioLand):
    """
    Enhanced Super Mario Land wrapper with Reinforcement Learning capabilities.

    This wrapper extends the original GameWrapperSuperMarioLand with:
    - RL-specific action space and reward calculation
    - State observation and serialization
    - Performance tracking and metrics
    - Episode management

    Features:
    - Action space: Movement, jumping, running with frame-perfect control
    - Reward shaping: Progress, coins, enemies defeated, time bonus, survival
    - State features: Mario position, velocity, level progress, enemies on screen
    - Performance metrics: Episode length, success rate, score progression
    """

    cartridge_title = "SUPER MARIOLAND"

    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        GameWrapperSuperMarioLand.__init__(self, *args, **kwargs)
        RLGameWrapperBase.__init__(self, *args, **kwargs)

        # Mario-specific RL attributes
        self.last_x_position = 0
        self.last_score = 0
        self.last_coins = 0
        self.last_time_left = 400
        self.max_x_position = 0
        self.enemies_defeated = 0
        self.powerups_collected = 0
        self.death_positions = []

        # Enhanced observation settings
        self.observation_type = kwargs.get('observation_type', 'tiles')  # 'tiles', 'pixels', 'hybrid'
        self.frame_stack_size = kwargs.get('frame_stack_size', 4)
        self.frame_stack = []

    def _define_action_space(self) -> List[RLAction]:
        """Define Mario action space."""
        return [
            RLAction("no_op", "right", 1),  # No action (hold right)
            RLAction("jump", "a", 1),  # Jump
            RLAction("run", "b", 1),  # Run/Sprint
            RLAction("jump_run", "a", 1),  # Jump while running
            RLAction("left", "left", 1),  # Move left
            RLAction("right", "right", 1),  # Move right
            RLAction("jump_left", "a", 1),  # Jump while moving left
            RLAction("jump_right", "a", 1),  # Jump while moving right
            RLAction("run_left", "b", 1),  # Run while moving left
            RLAction("run_right", "b", 1),  # Run while moving right
            RLAction("run_jump_left", "a", 1),  # Run and jump left
            RLAction("run_jump_right", "a", 1),  # Run and jump right
        ]

    def _define_reward_weights(self) -> Dict[str, float]:
        """Define reward weights for different events."""
        return {
            'progress': 1.0,      # Moving rightward
            'coin': 10.0,         # Collecting coins
            'enemy': 50.0,        # Defeating enemies
            'powerup': 25.0,      # Collecting powerups
            'death': -100.0,      # Mario dying
            'time': 0.01,         # Survival bonus per frame
            'level_complete': 500.0,  # Completing a level
            'score': 0.1,         # Score increase
            'backward': -0.5,     # Moving left (discouraged)
            'stuck': -0.1,        # Being stuck penalty
        }

    def _get_game_state(self) -> Dict[str, Any]:
        """Extract Mario game state."""
        state = {
            'world': self.world,
            'score': self.score,
            'coins': self.coins,
            'lives_left': self.lives_left,
            'time_left': self.time_left,
            'level_progress': self.level_progress,
            'mario_x': self.pyboy.memory[ADDR_MARIO_X],
            'mario_y': self.pyboy.memory[ADDR_MARIO_Y],
            'mario_velocity_x': self.pyboy.memory[ADDR_MARIO_VELOCITY_X],
            'mario_velocity_y': self.pyboy.memory[ADDR_MARIO_VELOCITY_Y],
            'mario_state': self.pyboy.memory[ADDR_MARIO_STATE],
            'enemies_defeated': self.enemies_defeated,
            'powerups_collected': self.powerups_collected,
            'max_x_position': self.max_x_position,
            'is_game_over': self.game_over(),
        }
        return state

    def _get_observation(self) -> np.ndarray:
        """Get observation based on configured type."""
        if self.observation_type == 'tiles':
            return self._get_tile_observation()
        elif self.observation_type == 'pixels':
            return self._get_pixel_observation()
        elif self.observation_type == 'hybrid':
            return self._get_hybrid_observation()
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")

    def _get_tile_observation(self) -> np.ndarray:
        """Get tile-based observation."""
        # Use the compressed mapping for RL
        self.game_area_mapping(mapping_compressed, 0)
        tiles = self.game_area()

        # Stack with additional state info
        state_vector = np.array([
            self.pyboy.memory[ADDR_MARIO_X] / 255.0,
            self.pyboy.memory[ADDR_MARIO_Y] / 255.0,
            self.pyboy.memory[ADDR_MARIO_VELOCITY_X] / 127.0,
            self.pyboy.memory[ADDR_MARIO_VELOCITY_Y] / 127.0,
            self.time_left / 400.0,
            self.lives_left / 3.0,
        ])

        # Flatten tiles and concatenate with state
        obs = np.concatenate([tiles.flatten(), state_vector])
        return obs.astype(np.float32)

    def _get_pixel_observation(self) -> np.ndarray:
        """Get raw pixel observation."""
        screen = self.pyboy.screen.ndarray
        # Convert to grayscale and resize if needed
        if len(screen.shape) == 3:
            screen = np.mean(screen, axis=2)
        # Normalize to [0, 1]
        obs = screen.astype(np.float32) / 255.0
        return obs

    def _get_hybrid_observation(self) -> np.ndarray:
        """Get hybrid observation combining tiles and pixels."""
        tile_obs = self._get_tile_observation()
        pixel_obs = self._get_pixel_observation()
        # Resize pixel obs to match and concatenate
        return np.concatenate([tile_obs, pixel_obs.flatten()[:len(tile_obs)]])

    def _calculate_reward(self, prev_state: Optional[Dict], curr_state: Dict) -> float:
        """Calculate reward based on state changes."""
        if prev_state is None:
            return 0.0

        reward = 0.0
        weights = self.reward_weights

        # Progress reward (moving right)
        x_progress = curr_state['mario_x'] - prev_state['mario_x']
        if x_progress > 0:
            reward += x_progress * weights['progress']
        elif x_progress < 0:
            reward += x_progress * weights['backward']  # Penalize going backward

        # Update max position
        if curr_state['mario_x'] > self.max_x_position:
            self.max_x_position = curr_state['mario_x']
            reward += weights['progress'] * 5  # Bonus for new max position

        # Coin collection
        coin_diff = curr_state['coins'] - prev_state['coins']
        if coin_diff > 0:
            reward += coin_diff * weights['coin']

        # Score increase
        score_diff = curr_state['score'] - prev_state['score']
        if score_diff > 0:
            reward += score_diff * weights['score']

        # Time bonus (survival)
        if curr_state['time_left'] > 0:
            reward += weights['time']

        # Death penalty
        if curr_state['is_game_over'] and not prev_state['is_game_over']:
            reward += weights['death']
            self.death_positions.append(curr_state['mario_x'])

        # Level completion detection (simplified)
        if curr_state['level_progress'] > prev_state['level_progress'] + 100:
            reward += weights['level_complete']

        # Stuck penalty
        if abs(x_progress) < 1 and curr_state['mario_y'] == prev_state['mario_y']:
            reward += weights['stuck']

        # Enemy defeat detection (simplified - check score jumps)
        if score_diff > 100:  # Likely from defeating an enemy
            self.enemies_defeated += 1
            reward += weights['enemy']

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Game over conditions
        if self.game_over():
            return True

        # Time up
        if self.time_left <= 0:
            return True

        # Out of lives
        if self.lives_left <= 0:
            return True

        # Optional: Max episode length
        max_steps = 18000  # 5 minutes at 60 FPS
        if self.step_count >= max_steps:
            return True

        return False

    def _get_action_mask(self) -> Optional[np.ndarray]:
        """Get action mask for valid actions."""
        mask = np.ones(len(self.action_space), dtype=np.bool_)

        # Some actions might not be valid in certain states
        mario_y = self.pyboy.memory[ADDR_MARIO_Y]
        mario_state = self.pyboy.memory[ADDR_MARIO_STATE]

        # Can't jump if already jumping (simplified check)
        if mario_y < 100:  # In the air
            # Disable jump actions when in air
            for i, action in enumerate(self.action_space):
                if 'jump' in action.name:
                    mask[i] = False

        # Disable actions when game over
        if self.game_over():
            mask[:] = False
            mask[0] = True  # Only allow no_op

        return mask

    def post_tick(self):
        """Update game state after each tick."""
        # Update parent game wrapper state
        GameWrapperSuperMarioLand.post_tick(self)

        # Update RL-specific tracking
        current_x = self.pyboy.memory[ADDR_MARIO_X]
        if current_x > self.max_x_position:
            self.max_x_position = current_x

        # Track score and coin changes for reward calculation
        if hasattr(self, 'last_score'):
            score_diff = self.score - self.last_score
            if score_diff > 100:  # Likely enemy defeat
                self.enemies_defeated += 1

        if hasattr(self, 'last_coins'):
            coin_diff = self.coins - self.last_coins
            if coin_diff > 0:  # Coin collected
                self.powerups_collected += coin_diff

        self.last_score = self.score
        self.last_coins = self.coins
        self.last_x_position = current_x

    def start_game(self, timer_div=None, world_level=None, unlock_level_select=False):
        """Start the game with RL-specific initialization."""
        # Reset RL tracking variables
        self.last_x_position = 0
        self.last_score = 0
        self.last_coins = 0
        self.max_x_position = 0
        self.enemies_defeated = 0
        self.powerups_collected = 0
        self.death_positions = []
        self.frame_stack = []

        # Call parent start_game
        GameWrapperSuperMarioLand.start_game(
            self, timer_div=timer_div,
            world_level=world_level,
            unlock_level_select=unlock_level_select
        )

    def get_learning_progression(self) -> Dict[str, Any]:
        """Get detailed learning progression statistics."""
        if not self.metrics_history:
            return {}

        recent_episodes = self.metrics_history[-20:]  # Last 20 episodes

        # Calculate skill progression
        max_positions = []
        completion_rates = []
        death_positions_analysis = []

        for metrics in recent_episodes:
            max_positions.append(metrics.additional_metrics.get('max_x_position', 0))
            if metrics.length > 100:  # Considered a success
                completion_rates.append(1.0)
            else:
                completion_rates.append(0.0)

        progression = {
            'skill_progression': {
                'avg_max_position': np.mean(max_positions),
                'max_position_trend': np.polyfit(range(len(max_positions)), max_positions, 1)[0] if len(max_positions) > 1 else 0,
                'completion_rate_trend': np.polyfit(range(len(completion_rates)), completion_rates, 1)[0] if len(completion_rates) > 1 else 0,
            },
            'death_analysis': {
                'avg_death_position': np.mean(self.death_positions) if self.death_positions else 0,
                'death_position_std': np.std(self.death_positions) if self.death_positions else 0,
                'most_common_death_position': np.bincount(self.death_positions).argmax() if self.death_positions else 0,
            },
            'enemy_defeat_rate': np.mean([m.additional_metrics.get('enemies_defeated', 0) for m in recent_episodes]),
            'coin_collection_rate': np.mean([m.additional_metrics.get('coins', 0) for m in recent_episodes]),
        }

        return progression

    def __repr__(self):
        """Enhanced representation with RL stats."""
        base_repr = GameWrapperSuperMarioLand.__repr__(self)
        performance = self.get_performance_summary()
        progression = self.get_learning_progression()

        if performance:
            rl_info = (
                f"\nRL Stats:\n"
                f"  Episodes: {performance['total_episodes']}\n"
                f"  Recent Avg Reward: {performance['recent_avg_reward']:.2f}\n"
                f"  Recent Success Rate: {performance['recent_success_rate']:.2%}\n"
                f"  Best Reward: {performance['best_reward']:.2f}\n"
                f"  Current Episode: {self.episode_count} (Step: {self.step_count})\n"
                f"  Max Position: {self.max_x_position}\n"
                f"  Enemies Defeated: {self.enemies_defeated}\n"
            )

            if progression:
                skill = progression['skill_progression']
                rl_info += (
                    f"  Position Trend: {skill['max_position_trend']:.2f}\n"
                    f"  Success Trend: {skill['completion_rate_trend']:.3f}\n"
                )

            return base_repr + rl_info

        return base_repr + f"\nRL Stats: No episodes completed yet"