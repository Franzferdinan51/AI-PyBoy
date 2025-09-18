#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "RLGameWrapperTetris.cartridge_title": False,
    "RLGameWrapperTetris.post_tick": False,
}

import numpy as np
from typing import Dict, List, Optional, Any

import pyboy
from pyboy.api.constants import TILES

from .game_wrapper_tetris import GameWrapperTetris, mapping_minimal, mapping_compressed
from .rl_game_wrapper_base import RLGameWrapperBase, RLAction, RLState

logger = pyboy.logging.get_logger(__name__)

# Tetromino types and shapes
TETROMINOES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'J': [[1, 0, 0], [1, 1, 1]],
    'L': [[0, 0, 1], [1, 1, 1]]
}

# Tetris memory addresses
ADDR_CURRENT_PIECE_X = 0xC070
ADDR_CURRENT_PIECE_Y = 0xC071
ADDR_CURRENT_PIECE_TYPE = 0xC072
ADDR_CURRENT_PIECE_ROTATION = 0xC073
ADDR_NEXT_PIECE = 0xC213
ADDR_SCORE = 0xC0B8
ADDR_LINES = 0xC0BE
ADDR_LEVEL = 0xC0C4
ADDR_GAME_OVER = 0xC0A4

class RLGameWrapperTetris(RLGameWrapperBase, GameWrapperTetris):
    """
    Enhanced Tetris wrapper with Reinforcement Learning capabilities.

    This wrapper extends the original GameWrapperTetris with:
    - Advanced action space for piece placement and rotation
    - Sophisticated reward calculation for line clearing and strategy
    - Board state analysis with heuristics (holes, height, bumps)
    - Performance tracking for skill progression

    Features:
    - Action space: Movement, rotation, drop with strategic timing
    - Reward shaping: Lines cleared, combo bonuses, strategic placement
    - State features: Board configuration, current piece, next piece, holes
    - Performance metrics: Lines per minute, level progression, survival time
    """

    cartridge_title = "TETRIS"

    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        GameWrapperTetris.__init__(self, *args, **kwargs)
        RLGameWrapperBase.__init__(self, *args, **kwargs)

        # Tetris-specific RL attributes
        self.last_lines = 0
        self.last_score = 0
        self.last_level = 0
        self.lines_cleared_this_episode = 0
        self.max_height = 0
        self.total_holes = 0
        self.combo_count = 0
        self.pieces_placed = 0
        self.perfect_clears = 0
        self.tetris_count = 0  # 4-line clears

        # Enhanced observation settings
        self.observation_type = kwargs.get('observation_type', 'hybrid')  # 'tiles', 'hybrid', 'analysis'
        self.include_next_piece = kwargs.get('include_next_piece', True)
        self.include_heuristics = kwargs.get('include_heuristics', True)

        # Strategy parameters
        self.aggressive_mode = kwargs.get('aggressive_mode', False)
        self.survival_mode = kwargs.get('survival_mode', False)

    def _define_action_space(self) -> List[RLAction]:
        """Define Tetris action space."""
        return [
            # Movement
            RLAction("left", "left", 1),
            RLAction("right", "right", 1),
            RLAction("down", "down", 1),
            RLAction("rotate_left", "b", 1),  # B button rotates in Tetris
            RLAction("rotate_right", "a", 1),  # A button rotates in Tetris
            RLAction("hard_drop", "up", 1),   # Hard drop

            # Strategic movements (multiple frames)
            RLAction("fast_left", "left", 5),
            RLAction("fast_right", "right", 5),
            RLAction("soft_drop", "down", 3),
            RLAction("hold_piece", "select", 1),  # Some versions have hold

            # No action (let piece fall)
            RLAction("wait", "start", 1),
        ]

    def _define_reward_weights(self) -> Dict[str, float]:
        """Define reward weights for different events."""
        base_weights = {
            'line_clear': 1.0,        # Base reward per line cleared
            'single': 1.0,           # 1 line cleared
            'double': 3.0,           # 2 lines cleared
            'triple': 5.0,           # 3 lines cleared
            'tetris': 8.0,           # 4 lines cleared (Tetris!)
            'combo': 2.0,            # Combo bonus
            'perfect_clear': 50.0,    # Clear entire board
            'piece_placed': 0.1,      # Successfully placed piece
            'survival': 0.01,        # Survival per frame
            'level_up': 10.0,        # Level progression
            'game_over': -100.0,     # Game over penalty
            'hole_created': -0.5,     # Creating holes penalty
            'height_increase': -0.1, # Increasing max height penalty
            'bumpiness': -0.01,      # Uneven surface penalty
        }

        # Adjust weights based on mode
        if self.aggressive_mode:
            base_weights.update({
                'line_clear': 2.0,
                'tetris': 15.0,
                'perfect_clear': 100.0,
                'hole_created': -0.2,
                'height_increase': -0.05,
            })

        if self.survival_mode:
            base_weights.update({
                'survival': 0.05,
                'hole_created': -1.0,
                'height_increase': -0.5,
                'game_over': -200.0,
                'piece_placed': 0.05,
            })

        return base_weights

    def _get_game_state(self) -> Dict[str, Any]:
        """Extract Tetris game state."""
        # Get board state
        board = self._get_board_state()

        # Calculate heuristics
        holes, max_height, bumpiness = self._calculate_board_heuristics(board)

        state = {
            'score': self.score,
            'lines': self.lines,
            'level': self.level,
            'current_piece_x': self.pyboy.memory[ADDR_CURRENT_PIECE_X],
            'current_piece_y': self.pyboy.memory[ADDR_CURRENT_PIECE_Y],
            'current_piece_type': self.pyboy.memory[ADDR_CURRENT_PIECE_TYPE],
            'current_piece_rotation': self.pyboy.memory[ADDR_CURRENT_PIECE_ROTATION],
            'next_piece': self.pyboy.memory[ADDR_NEXT_PIECE],
            'is_game_over': self.game_over(),
            'holes': holes,
            'max_height': max_height,
            'bumpiness': bumpiness,
            'lines_cleared_this_episode': self.lines_cleared_this_episode,
            'pieces_placed': self.pieces_placed,
            'combo_count': self.combo_count,
            'perfect_clears': self.perfect_clears,
            'tetris_count': self.tetris_count,
        }

        return state

    def _get_observation(self) -> np.ndarray:
        """Get observation based on configured type."""
        if self.observation_type == 'tiles':
            return self._get_tile_observation()
        elif self.observation_type == 'hybrid':
            return self._get_hybrid_observation()
        elif self.observation_type == 'analysis':
            return self._get_analysis_observation()
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")

    def _get_board_state(self) -> np.ndarray:
        """Get the current board state as a binary matrix."""
        # Get game area
        tiles = self.game_area()

        # Convert to binary (0 = empty, 1 = filled)
        board = (tiles != 47).astype(np.int8)  # 47 is the blank tile

        return board

    def _get_tile_observation(self) -> np.ndarray:
        """Get tile-based observation."""
        # Use compressed mapping
        self.game_area_mapping(mapping_compressed, 0)
        tiles = self.game_area()

        # Add next piece information
        next_piece_one_hot = np.zeros(7)  # 7 piece types
        next_piece_idx = min(self.pyboy.memory[ADDR_NEXT_PIECE] // 4, 6)
        next_piece_one_hot[next_piece_idx] = 1

        # Add level info
        level_normalized = min(self.level / 20.0, 1.0)

        # Combine observations
        obs = np.concatenate([tiles.flatten(), next_piece_one_hot, [level_normalized]])

        return obs.astype(np.float32)

    def _get_hybrid_observation(self) -> np.ndarray:
        """Get hybrid observation combining board state and analysis."""
        board = self._get_board_state()
        state_vector = self._get_state_vector()

        return np.concatenate([board.flatten(), state_vector]).astype(np.float32)

    def _get_analysis_observation(self) -> np.ndarray:
        """Get analysis-based observation with heuristics."""
        board = self._get_board_state()
        holes, max_height, bumpiness = self._calculate_board_heuristics(board)

        # Current piece info
        current_piece = self._get_current_piece_info()

        # Next piece info
        next_piece = self._get_next_piece_info()

        # Analysis vector
        analysis_vector = np.array([
            holes / 50.0,  # Normalized holes
            max_height / 20.0,  # Normalized height
            bumpiness / 50.0,  # Normalized bumpiness
            self.lines / 200.0,  # Normalized lines
            self.level / 20.0,  # Normalized level
            self.lines_cleared_this_episode / 500.0,  # Episode progress
        ])

        # Combine all info
        obs = np.concatenate([board.flatten(), current_piece, next_piece, analysis_vector])

        return obs.astype(np.float32)

    def _get_state_vector(self) -> np.ndarray:
        """Get normalized state vector."""
        state = self._get_game_state()

        vector = np.array([
            state['score'] / 999999.0,
            state['lines'] / 200.0,
            state['level'] / 20.0,
            state['current_piece_x'] / 10.0,
            state['current_piece_y'] / 20.0,
            state['current_piece_type'] / 7.0,
            state['current_piece_rotation'] / 3.0,
            state['holes'] / 50.0,
            state['max_height'] / 20.0,
            state['bumpiness'] / 50.0,
            float(state['is_game_over']),
        ])

        return vector

    def _get_current_piece_info(self) -> np.ndarray:
        """Get current piece information as one-hot vector."""
        piece_type = min(self.pyboy.memory[ADDR_CURRENT_PIECE_TYPE] // 4, 6)
        rotation = min(self.pyboy.memory[ADDR_CURRENT_PIECE_ROTATION], 3)

        piece_one_hot = np.zeros(7)
        rotation_one_hot = np.zeros(4)
        piece_one_hot[piece_type] = 1
        rotation_one_hot[rotation] = 1

        return np.concatenate([piece_one_hot, rotation_one_hot])

    def _get_next_piece_info(self) -> np.ndarray:
        """Get next piece information as one-hot vector."""
        next_piece_idx = min(self.pyboy.memory[ADDR_NEXT_PIECE] // 4, 6)
        next_piece_one_hot = np.zeros(7)
        next_piece_one_hot[next_piece_idx] = 1

        return next_piece_one_hot

    def _calculate_board_heuristics(self, board: np.ndarray) -> tuple:
        """Calculate board heuristics: holes, max height, bumpiness."""
        holes = 0
        max_height = 0
        bumpiness = 0
        column_heights = []

        # Calculate column heights and holes
        for col in range(board.shape[1]):
            # Find height of this column
            height = 0
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    height = board.shape[0] - row
                    break
            column_heights.append(height)
            max_height = max(max_height, height)

            # Count holes in this column
            for row in range(board.shape[0] - height, board.shape[0]):
                if board[row, col] == 0 and row > board.shape[0] - height - 1:
                    holes += 1

        # Calculate bumpiness
        for i in range(len(column_heights) - 1):
            bumpiness += abs(column_heights[i] - column_heights[i + 1])

        return holes, max_height, bumpiness

    def _calculate_reward(self, prev_state: Optional[Dict], curr_state: Dict) -> float:
        """Calculate reward based on state changes."""
        if prev_state is None:
            return 0.0

        reward = 0.0
        weights = self.reward_weights

        # Line clearing rewards
        lines_diff = curr_state['lines'] - prev_state['lines']
        if lines_diff > 0:
            self.lines_cleared_this_episode += lines_diff

            if lines_diff == 1:
                reward += weights['single']
            elif lines_diff == 2:
                reward += weights['double']
            elif lines_diff == 3:
                reward += weights['triple']
            elif lines_diff == 4:
                reward += weights['tetris']
                self.tetris_count += 1

            reward += lines_diff * weights['line_clear']

            # Check for perfect clear
            if curr_state['holes'] == 0 and curr_state['max_height'] == 0:
                reward += weights['perfect_clear']
                self.perfect_clears += 1

            # Combo detection
            if lines_diff >= 2:
                self.combo_count += 1
                reward += self.combo_count * weights['combo']
            else:
                self.combo_count = 0

        # Score increase
        score_diff = curr_state['score'] - prev_state['score']
        if score_diff > 0:
            reward += score_diff * 0.001  # Scale down score rewards

        # Level progression
        level_diff = curr_state['level'] - prev_state['level']
        if level_diff > 0:
            reward += level_diff * weights['level_up']

        # Piece placement
        piece_x_diff = abs(curr_state['current_piece_x'] - prev_state['current_piece_x'])
        piece_y_diff = abs(curr_state['current_piece_y'] - prev_state['current_piece_y'])
        if piece_y_diff > 5:  # Piece has been placed (moved down significantly)
            reward += weights['piece_placed']
            self.pieces_placed += 1

        # Board heuristics penalties
        holes_diff = curr_state['holes'] - prev_state['holes']
        if holes_diff > 0:
            reward += holes_diff * weights['hole_created']

        height_diff = curr_state['max_height'] - prev_state['max_height']
        if height_diff > 0:
            reward += height_diff * weights['height_increase']

        bumpiness_diff = curr_state['bumpiness'] - prev_state['bumpiness']
        if bumpiness_diff > 0:
            reward += bumpiness_diff * weights['bumpiness']

        # Survival bonus
        if not curr_state['is_game_over']:
            reward += weights['survival']

        # Game over penalty
        if curr_state['is_game_over'] and not prev_state['is_game_over']:
            reward += weights['game_over']

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        if self.game_over():
            return True

        # Optional: Max pieces placed for training
        max_pieces = 1000
        if self.pieces_placed >= max_pieces:
            return True

        # Optional: Max lines cleared
        max_lines = 500
        if self.lines_cleared_this_episode >= max_lines:
            return True

        return False

    def _get_action_mask(self) -> Optional[np.ndarray]:
        """Get action mask for valid actions."""
        mask = np.ones(len(self.action_space), dtype=np.bool_)

        # Can't move if at edges
        piece_x = self.pyboy.memory[ADDR_CURRENT_PIECE_X]
        if piece_x <= 0:
            mask[0] = False  # Can't move left
            mask[6] = False  # Can't fast left
        if piece_x >= 9:
            mask[1] = False  # Can't move right
            mask[7] = False  # Can't fast right

        # Can't drop if at bottom
        piece_y = self.pyboy.memory[ADDR_CURRENT_PIECE_Y]
        if piece_y >= 17:
            mask[2] = False  # Can't move down
            mask[8] = False  # Can't soft drop
            mask[5] = False  # Can't hard drop

        # Disable actions when game over
        if self.game_over():
            mask[:] = False
            mask[9] = True  # Only allow wait

        return mask

    def post_tick(self):
        """Update game state after each tick."""
        # Update parent game wrapper state
        GameWrapperTetris.post_tick(self)

        # Update tracking variables
        board = self._get_board_state()
        holes, max_height, bumpiness = self._calculate_board_heuristics(board)

        self.max_height = max(self.max_height, max_height)
        self.total_holes = holes

        # Update last values
        self.last_lines = self.lines
        self.last_score = self.score
        self.last_level = self.level

    def start_game(self, timer_div=None):
        """Start the game with RL-specific initialization."""
        # Reset RL tracking variables
        self.last_lines = 0
        self.last_score = 0
        self.last_level = 0
        self.lines_cleared_this_episode = 0
        self.max_height = 0
        self.total_holes = 0
        self.combo_count = 0
        self.pieces_placed = 0
        self.perfect_clears = 0
        self.tetris_count = 0

        # Call parent start_game
        GameWrapperTetris.start_game(self, timer_div=timer_div)

    def get_tetris_statistics(self) -> Dict[str, float]:
        """Get detailed Tetris performance statistics."""
        if self.pieces_placed == 0:
            return {}

        return {
            'lines_per_piece': self.lines_cleared_this_episode / max(self.pieces_placed, 1),
            'tetris_rate': self.tetris_count / max(self.pieces_placed, 1),
            'perfect_clear_rate': self.perfect_clears / max(self.pieces_placed, 1),
            'avg_holes_per_piece': self.total_holes / max(self.pieces_placed, 1),
            'max_height_ratio': self.max_height / 20.0,
            'combo_rate': self.combo_count / max(self.pieces_placed, 1),
            'level_progression': self.level / 20.0,
        }

    def get_skill_assessment(self) -> Dict[str, str]:
        """Get skill assessment based on performance."""
        stats = self.get_tetris_statistics()

        if not stats:
            return {'skill_level': 'Beginner', 'assessment': 'No data yet'}

        # Skill assessment logic
        skill_level = 'Beginner'
        assessment = []

        lpp = stats['lines_per_piece']
        tetris_rate = stats['tetris_rate']
        avg_holes = stats['avg_holes_per_piece']

        if lpp > 0.8:
            skill_level = 'Expert'
            assessment.append('Excellent line clearing efficiency')
        elif lpp > 0.6:
            skill_level = 'Advanced'
            assessment.append('Good line clearing efficiency')
        elif lpp > 0.4:
            skill_level = 'Intermediate'
            assessment.append('Decent line clearing efficiency')

        if tetris_rate > 0.05:
            assessment.append('Good Tetris rate')
        elif tetris_rate > 0.02:
            assessment.append('Occasional Tetris clears')

        if avg_holes < 0.1:
            assessment.append('Excellent board control')
        elif avg_holes < 0.3:
            assessment.append('Good board control')
        elif avg_holes > 0.5:
            assessment.append('Need to reduce holes')

        return {
            'skill_level': skill_level,
            'assessment': ', '.join(assessment) if assessment else 'Keep practicing!',
        }

    def __repr__(self):
        """Enhanced representation with Tetris-specific RL stats."""
        base_repr = GameWrapperTetris.__repr__(self)
        performance = self.get_performance_summary()
        stats = self.get_tetris_statistics()
        skill = self.get_skill_assessment()

        if performance:
            rl_info = (
                f"\nTetris RL Stats:\n"
                f"  Episodes: {performance['total_episodes']}\n"
                f"  Recent Avg Reward: {performance['recent_avg_reward']:.2f}\n"
                f"  Success Rate: {performance['recent_success_rate']:.2%}\n"
                f"  Skill Level: {skill['skill_level']}\n"
                f"  Lines This Episode: {self.lines_cleared_this_episode}\n"
                f"  Pieces Placed: {self.pieces_placed}\n"
                f"  Tetrises: {self.tetris_count}\n"
                f"  Perfect Clears: {self.perfect_clears}\n"
                f"  Max Height: {self.max_height}\n"
            )

            if stats:
                rl_info += (
                    f"  Lines/Piece: {stats['lines_per_piece']:.2f}\n"
                    f"  Tetris Rate: {stats['tetris_rate']:.2%}\n"
                    f"  Avg Holes: {stats['avg_holes_per_piece']:.2f}\n"
                )

            return base_repr + rl_info

        return base_repr + f"\nTetris RL Stats: No episodes completed yet"