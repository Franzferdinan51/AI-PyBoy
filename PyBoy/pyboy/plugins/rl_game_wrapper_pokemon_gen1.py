#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "RLGameWrapperPokemonGen1.cartridge_title": False,
    "RLGameWrapperPokemonGen1.post_tick": False,
}

import numpy as np
from typing import Dict, List, Optional, Any

import pyboy
from pyboy.api.constants import TILES

from .game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
from .rl_game_wrapper_base import RLGameWrapperBase, RLAction, RLState

logger = pyboy.logging.get_logger(__name__)

# Pokemon Gen 1 memory addresses
ADDR_PLAYER_X = 0xD362
ADDR_PLAYER_Y = 0xD361
ADDR_PLAYER_DIR = 0xD364
ADDR_MAP_GROUP = 0xD35E
ADDR_MAP_NUMBER = 0xD35F
ADDR_MONEY = 0xD347
ADDR_BADGES = 0xD356
ADDR_POKEMON_COUNT = 0xD163
ADDR_CURRENT_POKEMON_HP = 0xD18C
ADDR_CURRENT_POKEMON_MAX_HP = 0xD18E
ADDR_CURRENT_POKEMON_LEVEL = 0xD18C
ADDR_CURRENT_POKEMON_EXP = 0xD18E
ADDR_BATTLE_STATUS = 0xD057
ADDR_DIALOGUE_STATE = 0xC4A5
ADDR_PLAYER_STATE = 0xC109

# Player states
PLAYER_STATE_NORMAL = 0x00
PLAYER_STATE_BATTLE = 0x01
PLAYER_STATE_MENU = 0x02
PLAYER_STATE_DIALOGUE = 0x03

# Battle states
BATTLE_STATE_ACTIVE = 0x01
BATTLE_STATE_ENEMY_FAINTED = 0x02
BATTLE_STATE_PLAYER_FAINTED = 0x04
BATTLE_STATE_CAUGHT = 0x08

class RLGameWrapperPokemonGen1(RLGameWrapperBase, GameWrapperPokemonGen1):
    """
    Enhanced Pokemon Gen 1 wrapper with Reinforcement Learning capabilities.

    This wrapper extends the original GameWrapperPokemonGen1 with:
    - RL-specific action space for exploration and battles
    - Complex reward calculation for RPG progression
    - State observation including inventory, party, and battle status
    - Performance tracking for completionist goals

    Features:
    - Action space: Movement, menu navigation, battle actions
    - Reward shaping: EXP gain, Pokemon caught, badges earned, exploration
    - State features: Player position, inventory, party status, battle state
    - Performance metrics: Pokedex completion, gym badges, playthrough speed
    """

    cartridge_title = None  # Will match POKEMON RED or BLUE

    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        GameWrapperPokemonGen1.__init__(self, *args, **kwargs)
        RLGameWrapperBase.__init__(self, *args, **kwargs)

        # Pokemon-specific RL attributes
        self.last_exp = 0
        self.last_money = 0
        self.last_badges = 0
        self.last_pokemon_count = 0
        self.visited_maps = set()
        self.wild_pokemon_encountered = 0
        self.pokemon_caught = 0
        self.trainers_defeated = 0
        self.total_exp_gained = 0

        # Enhanced observation settings
        self.observation_type = kwargs.get('observation_type', 'tiles')  # 'tiles', 'hybrid', 'minimal'
        self.include_battle_state = kwargs.get('include_battle_state', True)

        # Track game progression milestones
        self.milestones = {
            'first_pokemon_caught': False,
            'first_badge': False,
            'first_evolution': False,
            'entered_victory_road': False,
            'defeated_elite_four': False,
        }

    def enabled(self):
        """Check if this wrapper is enabled for the current game."""
        return (self.pyboy.cartridge_title == "POKEMON RED") or (self.pyboy.cartridge_title == "POKEMON BLUE")

    def _define_action_space(self) -> List[RLAction]:
        """Define Pokemon action space."""
        return [
            # Movement
            RLAction("up", "up", 1),
            RLAction("down", "down", 1),
            RLAction("left", "left", 1),
            RLAction("right", "right", 1),

            # Actions
            RLAction("a_button", "a", 1),
            RLAction("b_button", "b", 1),
            RLAction("start", "start", 1),
            RLAction("select", "select", 1),

            # Direction combinations (for movement)
            RLAction("up_left", "up", 1),  # Will be combined with left
            RLAction("up_right", "up", 1),
            RLAction("down_left", "down", 1),
            RLAction("down_right", "down", 1),

            # Quick actions (multiple frame actions)
            RLAction("talk_interact", "a", 5),  # Longer press for talking
            RLAction("menu_cancel", "b", 3),  # Cancel menu
        ]

    def _define_reward_weights(self) -> Dict[str, float]:
        """Define reward weights for different events."""
        return {
            'exp_gain': 1.0,           # Experience points gained
            'pokemon_caught': 100.0,    # Catching a new Pokemon
            'badge_earned': 500.0,      # Earning a gym badge
            'money_gained': 0.1,        # Money earned
            'trainer_defeated': 50.0,    # Defeating a trainer
            'wild_pokemon_defeated': 10.0,  # Defeating wild Pokemon
            'new_area': 20.0,          # Exploring new areas
            'evolution': 200.0,         # Pokemon evolution
            'pokedex_entry': 150.0,     # New Pokedex entry
            'item_found': 25.0,         # Finding items
            'survival': 0.01,           # Survival bonus per step
            'blackout': -200.0,         # Player blackout (lost battle)
            'poisoned': -0.1,           # Being poisoned penalty
            'stuck': -0.5,              # Being stuck penalty
        }

    def _get_game_state(self) -> Dict[str, Any]:
        """Extract Pokemon game state."""
        # Get current Pokemon stats
        current_pokemon_hp = self._read_memory(ADDR_CURRENT_POKEMON_HP, 2)
        current_pokemon_max_hp = self._read_memory(ADDR_CURRENT_POKEMON_MAX_HP, 2)
        current_pokemon_level = self.pyboy.memory[ADDR_CURRENT_POKEMON_LEVEL]
        current_pokemon_exp = self._read_memory(ADDR_CURRENT_POKEMON_EXP, 3)

        # Player status
        player_state = self.pyboy.memory[ADDR_PLAYER_STATE]
        battle_status = self.pyboy.memory[ADDR_BATTLE_STATUS]

        state = {
            'player_x': self.pyboy.memory[ADDR_PLAYER_X],
            'player_y': self.pyboy.memory[ADDR_PLAYER_Y],
            'player_direction': self.pyboy.memory[ADDR_PLAYER_DIR],
            'map_group': self.pyboy.memory[ADDR_MAP_GROUP],
            'map_number': self.pyboy.memory[ADDR_MAP_NUMBER],
            'money': self._bcd_to_money(self._read_memory(ADDR_MONEY, 3)),
            'badges': self.pyboy.memory[ADDR_BADGES],
            'pokemon_count': self.pyboy.memory[ADDR_POKEMON_COUNT],
            'current_pokemon_hp': current_pokemon_hp,
            'current_pokemon_max_hp': current_pokemon_max_hp,
            'current_pokemon_level': current_pokemon_level,
            'current_pokemon_exp': current_pokemon_exp,
            'player_state': player_state,
            'battle_status': battle_status,
            'in_battle': (player_state == PLAYER_STATE_BATTLE),
            'in_menu': (player_state == PLAYER_STATE_MENU),
            'in_dialogue': (player_state == PLAYER_STATE_DIALOGUE),
            'wild_pokemon_encountered': self.wild_pokemon_encountered,
            'pokemon_caught': self.pokemon_caught,
            'trainers_defeated': self.trainers_defeated,
            'total_exp_gained': self.total_exp_gained,
            'visited_maps_count': len(self.visited_maps),
        }

        return state

    def _get_observation(self) -> np.ndarray:
        """Get observation based on configured type."""
        if self.observation_type == 'tiles':
            return self._get_tile_observation()
        elif self.observation_type == 'hybrid':
            return self._get_hybrid_observation()
        elif self.observation_type == 'minimal':
            return self._get_minimal_observation()
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")

    def _get_tile_observation(self) -> np.ndarray:
        """Get tile-based observation with collision info."""
        # Get game area tiles
        tiles = self.game_area()

        # Get collision matrix
        collision = self.game_area_collision()

        # Get player position relative to view
        player_x = self.pyboy.memory[ADDR_PLAYER_X] % 20
        player_y = self.pyboy.memory[ADDR_PLAYER_Y] % 18

        # Create player position heatmap
        player_heatmap = np.zeros(tiles.shape)
        if 0 <= player_x < tiles.shape[1] and 0 <= player_y < tiles.shape[0]:
            player_heatmap[player_y, player_x] = 1

        # Combine observations
        obs = np.stack([tiles, collision, player_heatmap], axis=0)

        # Add state vector
        state_vector = self._get_state_vector()

        return np.concatenate([obs.flatten(), state_vector]).astype(np.float32)

    def _get_hybrid_observation(self) -> np.ndarray:
        """Get hybrid observation combining tiles and game state."""
        tiles = self.game_area()
        state_vector = self._get_state_vector()
        return np.concatenate([tiles.flatten(), state_vector]).astype(np.float32)

    def _get_minimal_observation(self) -> np.ndarray:
        """Get minimal observation with just essential state."""
        return self._get_state_vector().astype(np.float32)

    def _get_state_vector(self) -> np.ndarray:
        """Get normalized state vector."""
        state = self._get_game_state()

        vector = np.array([
            state['player_x'] / 255.0,
            state['player_y'] / 255.0,
            state['player_direction'] / 3.0,
            state['map_group'] / 255.0,
            state['map_number'] / 255.0,
            min(state['money'] / 999999.0, 1.0),  # Cap at max money
            state['badges'] / 8.0,
            state['pokemon_count'] / 6.0,
            state['current_pokemon_hp'] / max(state['current_pokemon_max_hp'], 1),
            state['current_pokemon_level'] / 100.0,
            float(state['in_battle']),
            float(state['in_menu']),
            float(state['in_dialogue']),
        ])

        return vector

    def _calculate_reward(self, prev_state: Optional[Dict], curr_state: Dict) -> float:
        """Calculate reward based on state changes."""
        if prev_state is None:
            return 0.0

        reward = 0.0
        weights = self.reward_weights

        # Experience gain
        exp_diff = curr_state['current_pokemon_exp'] - prev_state['current_pokemon_exp']
        if exp_diff > 0:
            reward += exp_diff * weights['exp_gain']
            self.total_exp_gained += exp_diff

        # Money gain
        money_diff = curr_state['money'] - prev_state['money']
        if money_diff > 0:
            reward += money_diff * weights['money_gained']

        # Badges earned
        badge_diff = curr_state['badges'] - prev_state['badges']
        if badge_diff > 0:
            reward += badge_diff * weights['badge_earned']
            if not self.milestones['first_badge']:
                self.milestones['first_badge'] = True
                reward += weights['badge_earned'] * 2  # Bonus for first badge

        # Pokemon count increase
        pokemon_diff = curr_state['pokemon_count'] - prev_state['pokemon_count']
        if pokemon_diff > 0:
            reward += pokemon_diff * weights['pokemon_caught']
            self.pokemon_caught += pokemon_diff

        # New area exploration
        current_map = (curr_state['map_group'], curr_state['map_number'])
        if current_map not in self.visited_maps:
            self.visited_maps.add(current_map)
            reward += weights['new_area']

        # Battle outcomes
        if curr_state['battle_status'] != prev_state['battle_status']:
            if curr_state['battle_status'] & BATTLE_STATE_ENEMY_FAINTED:
                reward += weights['wild_pokemon_defeated']
                self.wild_pokemon_encountered += 1
            elif curr_state['battle_status'] & BATTLE_STATE_CAUGHT:
                reward += weights['pokemon_caught']
            elif curr_state['battle_status'] & BATTLE_STATE_PLAYER_FAINTED:
                reward += weights['blackout']

        # Survival bonus
        reward += weights['survival']

        # Movement/position based rewards
        x_movement = abs(curr_state['player_x'] - prev_state['player_x'])
        y_movement = abs(curr_state['player_y'] - prev_state['player_y'])
        if x_movement + y_movement == 0:
            reward += weights['stuck']  # Penalty for not moving

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Pokemon is typically not "done" in the traditional sense
        # But we can define completion conditions

        # Check for game completion (defeated Elite Four)
        if self.milestones['defeated_elite_four']:
            return True

        # Optional: Max episode length for training
        max_steps = 100000  # Very long for RPG
        if self.step_count >= max_steps:
            return True

        # Check for complete Pokedex (all 151 Pokemon)
        if self.pokemon_caught >= 151:
            return True

        # Check for all badges
        current_state = self._get_game_state()
        if current_state['badges'] >= 8:
            return True

        return False

    def _get_action_mask(self) -> Optional[np.ndarray]:
        """Get action mask for valid actions."""
        mask = np.ones(len(self.action_space), dtype=np.bool_)

        state = self._get_game_state()

        # Different actions valid in different states
        if state['in_battle']:
            # In battle, only A, B, Start, Select are typically useful
            for i, action in enumerate(self.action_space):
                if 'up' in action.name or 'down' in action.name or 'left' in action.name or 'right' in action.name:
                    mask[i] = False
        elif state['in_menu']:
            # In menu, directional actions are valid
            pass
        elif state['in_dialogue']:
            # In dialogue, mostly A and B are useful
            for i, action in enumerate(self.action_space):
                if 'up' in action.name or 'down' in action.name or 'left' in action.name or 'right' in action.name:
                    if action.name not in ['talk_interact', 'menu_cancel']:
                        mask[i] = False

        return mask

    def _execute_action(self, action: RLAction):
        """Execute Pokemon-specific actions."""
        # For diagonal movements, we need to press both buttons
        if 'up_left' in action.name:
            self.pyboy.button_press('up')
            self.pyboy.button_press('left')
        elif 'up_right' in action.name:
            self.pyboy.button_press('up')
            self.pyboy.button_press('right')
        elif 'down_left' in action.name:
            self.pyboy.button_press('down')
            self.pyboy.button_press('left')
        elif 'down_right' in action.name:
            self.pyboy.button_press('down')
            self.pyboy.button_press('right')
        else:
            super()._execute_action(action)

    def post_tick(self):
        """Update game state after each tick."""
        # Update parent game wrapper state
        GameWrapperPokemonGen1.post_tick(self)

        # Update tracking variables
        current_state = self._get_game_state()

        # Update last values
        self.last_exp = current_state['current_pokemon_exp']
        self.last_money = current_state['money']
        self.last_badges = current_state['badges']
        self.last_pokemon_count = current_state['pokemon_count']

    def _read_memory(self, addr: int, size: int) -> int:
        """Read multi-byte memory value."""
        value = 0
        for i in range(size):
            value |= self.pyboy.memory[addr + i] << (i * 8)
        return value

    def _bcd_to_money(self, bcd_value: int) -> int:
        """Convert BCD money value to integer."""
        result = 0
        multiplier = 1
        while bcd_value > 0:
            digit = bcd_value & 0x0F
            result += digit * multiplier
            multiplier *= 10
            bcd_value >>= 4
        return result

    def get_pokedex_progress(self) -> Dict[str, float]:
        """Get Pokedex completion statistics."""
        # This is a simplified version - actual implementation would need to scan memory
        return {
            'pokemon_seen': 0,  # Would need to scan memory for seen flags
            'pokemon_caught': self.pokemon_caught,
            'completion_rate': self.pokemon_caught / 151.0,
            'rare_pokemon_caught': 0,  # Would need specific tracking
        }

    def get_gym_progress(self) -> Dict[str, Any]:
        """Get gym badge progress."""
        current_state = self._get_game_state()
        badges = current_state['badges']

        return {
            'badges_earned': badges,
            'badges_remaining': 8 - badges,
            'completion_rate': badges / 8.0,
            'recent_badge_gain': badges - self.last_badges if hasattr(self, 'last_badges') else 0,
        }

    def get_exploration_progress(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        return {
            'unique_maps_visited': len(self.visited_maps),
            'total_maps': 384,  # Approximate total maps in Gen 1
            'exploration_rate': len(self.visited_maps) / 384.0,
            'current_area': (self.pyboy.memory[ADDR_MAP_GROUP], self.pyboy.memory[ADDR_MAP_NUMBER]),
        }

    def __repr__(self):
        """Enhanced representation with Pokemon-specific RL stats."""
        base_repr = GameWrapperPokemonGen1.__repr__(self)
        performance = self.get_performance_summary()

        if performance:
            pokedex = self.get_pokedex_progress()
            gym = self.get_gym_progress()
            exploration = self.get_exploration_progress()

            rl_info = (
                f"\nPokemon RL Stats:\n"
                f"  Episodes: {performance['total_episodes']}\n"
                f"  Recent Avg Reward: {performance['recent_avg_reward']:.2f}\n"
                f"  Pokedex: {pokedex['pokemon_caught']}/151 ({pokedex['completion_rate']:.1%})\n"
                f"  Badges: {gym['badges_earned']}/8 ({gym['completion_rate']:.1%})\n"
                f"  Maps Explored: {exploration['unique_maps_visited']}/{exploration['total_maps']} ({exploration['exploration_rate']:.1%})\n"
                f"  Total EXP Gained: {self.total_exp_gained}\n"
                f"  Pokemon Caught: {self.pokemon_caught}\n"
                f"  Trainers Defeated: {self.trainers_defeated}\n"
            )

            return base_repr + rl_info

        return base_repr + f"\nPokemon RL Stats: No episodes completed yet"