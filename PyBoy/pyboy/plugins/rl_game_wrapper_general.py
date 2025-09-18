#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "RLGameWrapperGeneral.cartridge_title": False,
    "RLGameWrapperGeneral.post_tick": False,
}

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import time

import pyboy
from pyboy.api.constants import TILES, SPRITES
from pyboy.utils import PyBoyException

from .base_plugin import PyBoyGameWrapper
from .rl_game_wrapper_base import RLGameWrapperBase, RLAction, RLState

logger = pyboy.logging.get_logger(__name__)

class RLGameWrapperGeneral(RLGameWrapperBase):
    """
    General-purpose Game Boy game wrapper for Reinforcement Learning.

    This wrapper provides RL capabilities for any Game Boy game without requiring
    game-specific knowledge. It uses computer vision and pattern recognition to:
    - Detect game states and UI elements
    - Extract relevant features from screen pixels
    - Calculate rewards based on score changes and progress indicators
    - Provide flexible action spaces for different game types

    Features:
    - Universal action space for all Game Boy games
    - Computer vision-based state observation
    - Automatic reward detection from screen elements
    - Game type detection and adaptation
    - Performance tracking for unknown games
    """

    cartridge_title = None  # Works with any game

    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)

        # General game detection and analysis
        self.game_type = self._detect_game_type()
        self.score_region = self._detect_score_region()
        self.lives_region = self._detect_lives_region()
        self.game_over_patterns = self._detect_game_over_patterns()

        # Vision and analysis settings
        self.observation_type = kwargs.get('observation_type', 'hybrid')  # 'pixels', 'tiles', 'hybrid'
        self.vision_resolution = kwargs.get('vision_resolution', (84, 84))  # Downscaled resolution
        self.use_grayscale = kwargs.get('use_grayscale', True)
        self.use_frame_diff = kwargs.get('use_frame_diff', True)
        self.frame_stack_size = kwargs.get('frame_stack_size', 4)

        # Reward detection settings
        self.reward_threshold = kwargs.get('reward_threshold', 100)  # Minimum score change for reward
        self.reward_scaling = kwargs.get('reward_scaling', 0.01)  # Scale factor for rewards
        self.time_penalty = kwargs.get('time_penalty', -0.001)  # Penalty per frame to encourage speed
        self.death_penalty = kwargs.get('death_penalty', -10.0)

        # Tracking variables
        self.last_screen_hash = None
        self.last_score = 0
        self.stuck_counter = 0
        self.stuck_threshold = kwargs.get('stuck_threshold', 300)  # Frames to consider stuck
        self.visited_states = defaultdict(int)
        self.score_changes = []
        self.death_count = 0
        self.game_progress = 0.0

        # Frame stacking and difference
        self.frame_stack = []
        self.last_frame = None

        # Game-specific adaptations
        self.adaptive_rewards = kwargs.get('adaptive_rewards', True)
        self.learning_progress = {
            'score_progression': [],
            'survival_times': [],
            'exploration_rate': 0.1,
        }

    def _detect_game_type(self) -> str:
        """Detect the type of game based on visual patterns."""
        # This is a simplified detection - in practice would use more sophisticated CV
        try:
            # Get initial screen
            screen = self.pyboy.screen.ndarray

            # Count distinct colors (simple heuristic)
            if len(screen.shape) == 3:
                unique_colors = len(np.unique(screen.reshape(-1, screen.shape[2]), axis=0))
            else:
                unique_colors = len(np.unique(screen))

            # Check for platformer characteristics
            tile_pattern = self._analyze_tile_patterns()
            sprite_count = len(self._sprites_on_screen())

            if sprite_count > 5 and tile_pattern['horizontal_platforms'] > 0:
                return 'platformer'
            elif tile_pattern['grid_based'] > 0.8:
                return 'puzzle'
            elif unique_colors > 32:
                return 'action'
            else:
                return 'unknown'

        except Exception as e:
            logger.warning(f"Game type detection failed: {e}")
            return 'unknown'

    def _analyze_tile_patterns(self) -> Dict[str, float]:
        """Analyze tile patterns to determine game characteristics."""
        try:
            tiles = self.game_area()
            pattern_scores = {
                'horizontal_platforms': 0.0,
                'vertical_structures': 0.0,
                'grid_based': 0.0,
                'open_areas': 0.0,
            }

            height, width = tiles.shape

            # Check for horizontal platforms
            for y in range(height):
                continuous_row = 0
                for x in range(width):
                    if tiles[y, x] != 47:  # Non-empty tile
                        continuous_row += 1
                        if continuous_row >= 3:
                            pattern_scores['horizontal_platforms'] += 1
                            break

            # Check for vertical structures
            for x in range(width):
                continuous_col = 0
                for y in range(height):
                    if tiles[y, x] != 47:
                        continuous_col += 1
                        if continuous_col >= 3:
                            pattern_scores['vertical_structures'] += 1
                            break

            # Calculate grid-based score
            non_empty_tiles = np.sum(tiles != 47)
            total_tiles = height * width
            if total_tiles > 0:
                pattern_scores['grid_based'] = non_empty_tiles / total_tiles
                pattern_scores['open_areas'] = 1.0 - pattern_scores['grid_based']

            return pattern_scores

        except Exception as e:
            logger.warning(f"Tile pattern analysis failed: {e}")
            return {'horizontal_platforms': 0.0, 'vertical_structures': 0.0, 'grid_based': 0.0, 'open_areas': 0.0}

    def _detect_score_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Try to detect the score region on screen."""
        try:
            # Look for numeric patterns in typical score locations
            # Common score locations: top-left, top-right, bottom
            screen = self.pyboy.screen.ndarray

            # Check top 20 rows for numeric patterns
            for y in range(min(20, screen.shape[0])):
                for x in range(0, screen.shape[1] - 20, 5):
                    # Simple heuristic: look for areas with high variation (likely text)
                    region = screen[y:y+2, x:x+20]
                    if len(region.shape) == 3:
                        region = np.mean(region, axis=2)
                    variation = np.std(region)
                    if variation > 20:  # Threshold for detecting text
                        return (x, y, 20, 10)

            return None

        except Exception as e:
            logger.warning(f"Score region detection failed: {e}")
            return None

    def _detect_lives_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Try to detect the lives/health region on screen."""
        # Similar to score detection but focused on common life indicators
        try:
            screen = self.pyboy.screen.ndarray

            # Look for heart icons or numeric lives in typical locations
            for y in range(min(20, screen.shape[0])):
                for x in range(screen.shape[1] - 30, screen.shape[1] - 10):
                    region = screen[y:y+2, x:x+10]
                    if len(region.shape) == 3:
                        region = np.mean(region, axis=2)
                    # Look for bright areas (typical of life indicators)
                    if np.mean(region) > 200:
                        return (x, y, 10, 5)

            return None

        except Exception as e:
            logger.warning(f"Lives region detection failed: {e}")
            return None

    def _detect_game_over_patterns(self) -> List[str]:
        """Detect common game over patterns."""
        # These would be pre-defined patterns or learned patterns
        return [
            'GAME OVER',  # Text pattern
            'static_screen',  # No movement for extended time
            'black_screen',  # All black screen
            'score_zero_lives',  # Score shows 0 lives
        ]

    def _define_action_space(self) -> List[RLAction]:
        """Define general Game Boy action space."""
        base_actions = [
            # D-pad
            RLAction("up", "up", 1),
            RLAction("down", "down", 1),
            RLAction("left", "left", 1),
            RLAction("right", "right", 1),

            # Buttons
            RLAction("a", "a", 1),
            RLAction("b", "b", 1),
            RLAction("start", "start", 1),
            RLAction("select", "select", 1),
        ]

        # Add composite actions based on game type
        if self.game_type == 'platformer':
            base_actions.extend([
                RLAction("jump_right", "a", 1),  # Will combine with right
                RLAction("jump_left", "a", 1),   # Will combine with left
                RLAction("run_right", "b", 1),  # Will combine with right
            ])
        elif self.game_type == 'puzzle':
            base_actions.extend([
                RLAction("confirm", "a", 2),    # Longer press for menus
                RLAction("cancel", "b", 2),     # Longer press for cancel
            ])

        return base_actions

    def _define_reward_weights(self) -> Dict[str, float]:
        """Define reward weights based on game type."""
        base_weights = {
            'score_increase': 1.0,
            'progress_forward': 0.5,
            'survival': 0.01,
            'death_penalty': -10.0,
            'stuck_penalty': -0.1,
            'exploration_bonus': 0.05,
        }

        # Adjust based on game type
        if self.game_type == 'platformer':
            base_weights.update({
                'progress_forward': 1.0,
                'jump_success': 2.0,
                'enemy_defeat': 5.0,
            })
        elif self.game_type == 'puzzle':
            base_weights.update({
                'score_increase': 2.0,
                'move_efficiency': 0.5,
                'combo_bonus': 3.0,
            })
        elif self.game_type == 'action':
            base_weights.update({
                'enemy_defeat': 3.0,
                'item_collection': 2.0,
                'level_progress': 1.0,
            })

        return base_weights

    def _get_game_state(self) -> Dict[str, Any]:
        """Extract general game state."""
        # Get basic information available to any game
        frame_count = self.pyboy.frame_count

        # Extract score from detected region
        score = self._extract_score()

        # Extract lives/health from detected region
        lives = self._extract_lives()

        # Analyze screen for game state
        screen_hash = self._get_screen_hash()
        is_stuck = self._check_if_stuck(screen_hash)

        # Calculate exploration metrics
        novelty_score = self._calculate_novelty_score(screen_hash)

        state = {
            'frame_count': frame_count,
            'score': score,
            'lives': lives,
            'screen_hash': screen_hash,
            'is_stuck': is_stuck,
            'novelty_score': novelty_score,
            'game_type': self.game_type,
            'death_count': self.death_count,
            'visited_states_count': len(self.visited_states),
            'game_progress': self.game_progress,
        }

        return state

    def _extract_score(self) -> int:
        """Extract score from the detected score region."""
        if self.score_region is None:
            return 0

        try:
            x, y, w, h = self.score_region
            screen = self.pyboy.screen.ndarray

            if len(screen.shape) == 3:
                region = screen[y:y+h, x:x+w]
                # Convert to grayscale for OCR
                region = np.mean(region, axis=2)
            else:
                region = screen[y:y+h, x:x+w]

            # Simple score detection - look for bright areas (digits)
            brightness_threshold = 180
            digit_pixels = np.sum(region > brightness_threshold)

            # This is a very basic heuristic
            # In practice, you'd use OCR or more sophisticated pattern recognition
            estimated_score = digit_pixels * 10  # Rough estimate

            return estimated_score

        except Exception as e:
            logger.warning(f"Score extraction failed: {e}")
            return 0

    def _extract_lives(self) -> int:
        """Extract lives/health from the detected lives region."""
        if self.lives_region is None:
            return 3  # Default assumption

        try:
            x, y, w, h = self.lives_region
            screen = self.pyboy.screen.ndarray

            if len(screen.shape) == 3:
                region = screen[y:y+h, x:x+w]
                region = np.mean(region, axis=2)
            else:
                region = screen[y:y+h, x:x+w]

            # Count bright areas as lives
            brightness_threshold = 200
            life_indicators = np.sum(region > brightness_threshold)

            return max(1, life_indicators // 5)  # Rough estimate

        except Exception as e:
            logger.warning(f"Lives extraction failed: {e}")
            return 3

    def _get_screen_hash(self) -> int:
        """Get hash of current screen for state comparison."""
        try:
            screen = self.pyboy.screen.ndarray
            if len(screen.shape) == 3:
                screen = np.mean(screen, axis=2)

            # Downsample for faster hashing
            small_screen = cv2.resize(screen, (32, 24), interpolation=cv2.INTER_NEAREST)
            return hash(small_screen.tobytes())

        except Exception as e:
            logger.warning(f"Screen hashing failed: {e}")
            return 0

    def _check_if_stuck(self, current_hash: int) -> bool:
        """Check if the agent is stuck in a loop."""
        if self.last_screen_hash == current_hash:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_screen_hash = current_hash

        return self.stuck_counter > self.stuck_threshold

    def _calculate_novelty_score(self, screen_hash: int) -> float:
        """Calculate how novel the current state is."""
        self.visited_states[screen_hash] += 1
        total_visits = sum(self.visited_states.values())

        if total_visits == 0:
            return 1.0

        # Novelty is inverse of visitation frequency
        visit_frequency = self.visited_states[screen_hash] / total_visits
        return 1.0 - min(visit_frequency * 100, 1.0)

    def _get_observation(self) -> np.ndarray:
        """Get observation based on configured type."""
        if self.observation_type == 'pixels':
            return self._get_pixel_observation()
        elif self.observation_type == 'tiles':
            return self._get_tile_observation()
        elif self.observation_type == 'hybrid':
            return self._get_hybrid_observation()
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")

    def _get_pixel_observation(self) -> np.ndarray:
        """Get pixel-based observation with preprocessing."""
        # Get raw screen
        screen = self.pyboy.screen.ndarray

        # Convert to grayscale if requested
        if self.use_grayscale and len(screen.shape) == 3:
            screen = np.mean(screen, axis=2)

        # Resize to target resolution
        if screen.shape[:2] != self.vision_resolution:
            screen = cv2.resize(screen, self.vision_resolution, interpolation=cv2.INTER_NEAREST)

        # Normalize to [0, 1]
        obs = screen.astype(np.float32) / 255.0

        # Apply frame differencing if requested
        if self.use_frame_diff and self.last_frame is not None:
            frame_diff = obs - self.last_frame
            obs = np.concatenate([obs.flatten(), frame_diff.flatten()])
        else:
            obs = obs.flatten()

        # Update last frame
        self.last_frame = obs.copy()

        # Add to frame stack
        self.frame_stack.append(obs)
        if len(self.frame_stack) > self.frame_stack_size:
            self.frame_stack.pop(0)

        # Return stacked observation
        if len(self.frame_stack) == self.frame_stack_size:
            return np.concatenate(self.frame_stack)
        else:
            # Pad with zeros if not enough frames
            padding = np.zeros(self.frame_stack_size * len(obs) - len(obs) * len(self.frame_stack))
            return np.concatenate([obs] + [padding])

    def _get_tile_observation(self) -> np.ndarray:
        """Get tile-based observation."""
        tiles = self.game_area()

        # Add sprite information
        sprites = self._sprites_on_screen()
        sprite_info = np.zeros((len(sprites), 4))  # x, y, tile_id, on_screen

        for i, sprite in enumerate(sprites):
            sprite_info[i] = [sprite.x, sprite.y, sprite.tile_identifier, sprite.on_screen]

        # Flatten and combine
        obs = np.concatenate([tiles.flatten(), sprite_info.flatten()])

        return obs.astype(np.float32)

    def _get_hybrid_observation(self) -> np.ndarray:
        """Get hybrid observation combining pixels and game state."""
        pixel_obs = self._get_pixel_observation()[:1000]  # Limit size
        state_vector = self._get_state_vector()

        return np.concatenate([pixel_obs, state_vector]).astype(np.float32)

    def _get_state_vector(self) -> np.ndarray:
        """Get normalized state vector."""
        state = self._get_game_state()

        vector = np.array([
            state['score'] / 10000.0,  # Normalize score
            state['lives'] / 10.0,     # Normalize lives
            float(state['is_stuck']),
            state['novelty_score'],
            state['death_count'] / 10.0,
            state['visited_states_count'] / 1000.0,
            state['game_progress'],
        ])

        return vector

    def _calculate_reward(self, prev_state: Optional[Dict], curr_state: Dict) -> float:
        """Calculate reward based on state changes."""
        if prev_state is None:
            return 0.0

        reward = 0.0
        weights = self.reward_weights

        # Score-based reward
        score_diff = curr_state['score'] - prev_state['score']
        if score_diff > self.reward_threshold:
            reward += score_diff * weights['score_increase'] * self.reward_scaling
            self.score_changes.append(score_diff)

        # Progress reward (simplified)
        if curr_state['frame_count'] % 60 == 0:  # Check every second
            progress_reward = curr_state['novelty_score'] * weights['exploration_bonus']
            reward += progress_reward

        # Survival reward
        reward += weights['survival']

        # Stuck penalty
        if curr_state['is_stuck']:
            reward += weights['stuck_penalty']

        # Death detection
        if curr_state['lives'] < prev_state['lives']:
            self.death_count += 1
            reward += weights['death_penalty']

        # Time penalty (encourage efficiency)
        reward += self.time_penalty

        # Update game progress
        self.game_progress = min(1.0, curr_state['frame_count'] / 36000.0)  # 10 minutes

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        current_state = self._get_game_state()

        # Check for game over conditions
        if current_state['lives'] <= 0:
            return True

        # Check for extreme stuck condition
        if current_state['is_stuck'] and self.stuck_counter > self.stuck_threshold * 3:
            return True

        # Max episode length
        max_steps = 36000  # 10 minutes at 60 FPS
        if self.step_count >= max_steps:
            return True

        # Check for score progression requirement
        if self.step_count > 5000 and len(self.score_changes) == 0:
            return True  # No progress in reasonable time

        return False

    def _execute_action(self, action: RLAction):
        """Execute general game actions."""
        # Handle composite actions
        if action.name == "jump_right":
            self.pyboy.button_press("a")
            self.pyboy.button_press("right")
        elif action.name == "jump_left":
            self.pyboy.button_press("a")
            self.pyboy.button_press("left")
        elif action.name == "run_right":
            self.pyboy.button_press("b")
            self.pyboy.button_press("right")
        else:
            super()._execute_action(action)

    def post_tick(self):
        """Update game state after each tick."""
        # Update parent
        super().post_tick()

        # Update tracking
        current_state = self._get_game_state()
        self.last_score = current_state['score']

    def get_game_analysis(self) -> Dict[str, Any]:
        """Get analysis of the current game."""
        return {
            'game_type': self.game_type,
            'estimated_difficulty': self._estimate_difficulty(),
            'score_progression': np.mean(self.score_changes[-10:]) if self.score_changes else 0,
            'death_rate': self.death_count / max(self.episode_count, 1),
            'exploration_efficiency': len(self.visited_states) / max(self.step_count, 1),
            'stuck_frequency': sum(1 for s in self.visited_states.values() if s > 10) / max(len(self.visited_states), 1),
        }

    def _estimate_difficulty(self) -> str:
        """Estimate game difficulty based on agent performance."""
        if not self.metrics_history:
            return 'unknown'

        recent_episodes = self.metrics_history[-10:]
        avg_survival_time = np.mean([m.length for m in recent_episodes])

        if avg_survival_time < 1000:
            return 'very_hard'
        elif avg_survival_time < 3000:
            return 'hard'
        elif avg_survival_time < 6000:
            return 'medium'
        else:
            return 'easy'

    def adapt_rewards(self):
        """Adapt reward weights based on learning progress."""
        if not self.adaptive_rewards or len(self.metrics_history) < 5:
            return

        recent_performance = np.mean([m.total_reward for m in self.metrics_history[-5:]])
        baseline_performance = np.mean([m.total_reward for m in self.metrics_history[:-5]])

        if recent_performance > baseline_performance * 1.2:
            # Learning is going well, increase exploration
            self.reward_weights['exploration_bonus'] *= 1.1
            self.reward_weights['stuck_penalty'] *= 0.9
        elif recent_performance < baseline_performance * 0.8:
            # Learning is struggling, reduce penalties
            self.reward_weights['death_penalty'] *= 0.9
            self.reward_weights['stuck_penalty'] *= 0.9

    def __repr__(self):
        """Enhanced representation with general game RL stats."""
        performance = self.get_performance_summary()
        analysis = self.get_game_analysis()

        if performance:
            rl_info = (
                f"\nGeneral Game RL Stats:\n"
                f"  Game Type: {analysis['game_type']}\n"
                f"  Difficulty: {analysis['estimated_difficulty']}\n"
                f"  Episodes: {performance['total_episodes']}\n"
                f"  Recent Avg Reward: {performance['recent_avg_reward']:.2f}\n"
                f"  Success Rate: {performance['recent_success_rate']:.2%}\n"
                f"  Current Episode: {self.episode_count} (Step: {self.step_count})\n"
                f"  Deaths: {self.death_count}\n"
                f"  States Explored: {len(self.visited_states)}\n"
                f"  Avg Score Change: {analysis['score_progression']:.2f}\n"
                f"  Exploration Efficiency: {analysis['exploration_efficiency']:.3f}\n"
            )

            return f"General Game Boy Wrapper\n{rl_info}"

        return "General Game Boy Wrapper\nRL Stats: No episodes completed yet"