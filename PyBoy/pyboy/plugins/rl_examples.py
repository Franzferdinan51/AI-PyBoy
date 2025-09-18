#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
"""
RL Game Wrapper Examples

This file provides comprehensive examples of how to use the enhanced RL game wrappers
for reinforcement learning with Game Boy games.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional
import sys

# Add PyBoy to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import pyboy
    from pyboy import PyBoy
    from pyboy.plugins import (
        RLGameWrapperSuperMarioLand,
        RLGameWrapperTetris,
        RLGameWrapperPokemonGen1,
        RLGameWrapperGeneral
    )
    from pyboy.plugins.rl_visualization import (
        RLVisualizer, TrainingMetrics, create_visualization_suite
    )
    PYBOY_AVAILABLE = True
except ImportError as e:
    print(f"PyBoy not available: {e}")
    PYBOY_AVAILABLE = False


class RLAgentExample:
    """Example RL agent for demonstration purposes."""

    def __init__(self, action_space_size: int, learning_rate: float = 0.1):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.q_table = {}  # Simple Q-table for demonstration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)

        # Convert state to hashable key
        state_key = hash(state.tobytes())

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)

        return np.argmax(self.q_table[state_key])

    def update_q_value(self, state, action, reward, next_state, done: bool):
        """Update Q-value using Q-learning."""
        state_key = hash(state.tobytes())
        next_state_key = hash(next_state.tobytes())

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)

        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0

        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + 0.99 * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def example_super_mario_land():
    """Example using Super Mario Land RL wrapper."""
    if not PYBOY_AVAILABLE:
        print("PyBoy not available - skipping Super Mario Land example")
        return

    print("\n" + "="*60)
    print("SUPER MARIO LAND RL EXAMPLE")
    print("="*60)

    # Initialize PyBoy with Super Mario Land ROM
    # Note: You need to provide the actual ROM path
    rom_path = "path/to/super_mario_land.gb"
    if not os.path.exists(rom_path):
        print(f"ROM not found at {rom_path}")
        print("This is a demonstration - actual ROM file needed")
        return

    try:
        # Create PyBoy instance
        pyboy = PyBoy(
            rom_path,
            window_type="null",  # Headless for training
            game_wrapper=True   # Enable game wrapper
        )

        # Get the RL wrapper
        wrapper = pyboy.game_wrapper
        if not isinstance(wrapper, RLGameWrapperSuperMarioLand):
            print("RL wrapper not available - using standard wrapper")
            return

        # Initialize visualization
        viz = RLVisualizer("mario_rl_output", auto_save=True)

        # Initialize agent
        agent = RLAgentExample(wrapper.get_action_space_size())

        # Training parameters
        num_episodes = 100
        max_steps_per_episode = 18000  # 5 minutes at 60 FPS

        print(f"Starting training for {num_episodes} episodes...")
        print(f"Action space size: {wrapper.get_action_space_size()}")
        print(f"Available actions: {wrapper.get_action_names()}")

        for episode in range(num_episodes):
            # Reset environment
            state = wrapper.reset()
            total_reward = 0
            episode_start_time = time.time()

            print(f"Episode {episode + 1}/{num_episodes}")

            for step in range(max_steps_per_episode):
                # Get action from agent
                action = agent.get_action(state.observation)

                # Execute action
                next_state = wrapper.step(action)
                total_reward += next_state.reward

                # Update agent
                agent.update_q_value(
                    state.observation,
                    action,
                    next_state.reward,
                    next_state.observation,
                    next_state.done
                )

                state = next_state

                if next_state.done:
                    break

            # Calculate episode metrics
            episode_time = time.time() - episode_start_time
            success = total_reward > 100  # Arbitrary success threshold

            metrics = TrainingMetrics(
                episode=episode + 1,
                step=step + 1,
                reward=total_reward,
                length=step + 1,
                survival_time=episode_time,
                final_score=wrapper.score,
                success=success,
                game_specific_metrics={
                    'world': wrapper.world,
                    'coins': wrapper.coins,
                    'lives_left': wrapper.lives_left,
                    'level_progress': wrapper.level_progress,
                    'enemies_defeated': wrapper.enemies_defeated,
                },
                timestamp=time.time(),
                learning_rate=agent.learning_rate,
                epsilon=agent.epsilon
            )

            viz.add_metrics(metrics)

            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
                      f"Steps={step + 1}, Success={success}, "
                      f"Epsilon={agent.epsilon:.3f}")

        # Final summary
        print("\nTraining completed!")
        print(f"Final epsilon: {agent.epsilon:.3f}")
        print(f"Q-table size: {len(agent.q_table)} states")

        # Generate final visualizations
        viz.update_plots()
        viz.save_current_plots("mario_final_dashboard.png")

        if hasattr(viz, 'create_interactive_dashboard'):
            dashboard_path = viz.create_interactive_dashboard()
            if dashboard_path:
                print(f"Interactive dashboard saved: {dashboard_path}")

        # Generate performance report
        report = viz.generate_performance_report()
        print("\nPerformance Report:")
        print(report)

        # Save metrics
        viz.export_metrics("mario_training_metrics.json")

        pyboy.stop()

    except Exception as e:
        print(f"Error in Super Mario Land example: {e}")


def example_tetris():
    """Example using Tetris RL wrapper."""
    if not PYBOY_AVAILABLE:
        print("PyBoy not available - skipping Tetris example")
        return

    print("\n" + "="*60)
    print("TETRIS RL EXAMPLE")
    print("="*60)

    rom_path = "path/to/tetris.gb"
    if not os.path.exists(rom_path):
        print(f"ROM not found at {rom_path}")
        print("This is a demonstration - actual ROM file needed")
        return

    try:
        pyboy = PyBoy(
            rom_path,
            window_type="null",
            game_wrapper=True
        )

        wrapper = pyboy.game_wrapper
        if not isinstance(wrapper, RLGameWrapperTetris):
            print("RL wrapper not available - using standard wrapper")
            return

        # Create visualization suite
        viz_suite = create_visualization_suite("tetris_rl_output")

        # Initialize agent with Tetris-specific parameters
        agent = RLAgentExample(
            wrapper.get_action_space_size(),
            learning_rate=0.05  # Lower learning rate for Tetris
        )

        # Training parameters
        num_episodes = 50

        print(f"Starting Tetris training for {num_episodes} episodes...")
        print(f"Action space: {wrapper.get_action_names()}")

        for episode in range(num_episodes):
            state = wrapper.reset()
            total_reward = 0

            for step in range(10000):  # Shorter episodes for Tetris
                action = agent.get_action(state.observation)
                next_state = wrapper.step(action)
                total_reward += next_state.reward

                agent.update_q_value(
                    state.observation,
                    action,
                    next_state.reward,
                    next_state.observation,
                    next_state.done
                )

                state = next_state

                if next_state.done:
                    break

            # Get Tetris-specific statistics
            tetris_stats = wrapper.get_tetris_statistics()
            skill_assessment = wrapper.get_skill_assessment()

            metrics = TrainingMetrics(
                episode=episode + 1,
                step=step + 1,
                reward=total_reward,
                length=step + 1,
                survival_time=step / 60.0,
                final_score=wrapper.score,
                success=wrapper.lines > 50,  # Success if cleared 50+ lines
                game_specific_metrics={
                    'lines': wrapper.lines,
                    'level': wrapper.level,
                    'pieces_placed': wrapper.pieces_placed,
                    'tetrises': wrapper.tetris_count,
                    'perfect_clears': wrapper.perfect_clears,
                    **tetris_stats
                },
                timestamp=time.time(),
                learning_rate=agent.learning_rate,
                epsilon=agent.epsilon
            )

            viz_suite['visualizer'].add_metrics(metrics)
            viz_suite['monitor'].add_metric(
                episode + 1, total_reward, step + 1,
                lines=wrapper.lines,
                level=wrapper.level,
                skill_level=skill_assessment['skill_level']
            )

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: Lines={wrapper.lines}, "
                      f"Level={wrapper.level}, Tetrises={wrapper.tetris_count}, "
                      f"Skill={skill_assessment['skill_level']}")

        print(f"\nFinal Tetris Statistics:")
        print(f"Total lines cleared: {wrapper.lines}")
        print(f"Final level: {wrapper.level}")
        print(f"Tetrises achieved: {wrapper.tetris_count}")
        print(f"Perfect clears: {wrapper.perfect_clears}")

        pyboy.stop()

    except Exception as e:
        print(f"Error in Tetris example: {e}")


def example_pokemon():
    """Example using Pokemon Gen 1 RL wrapper."""
    if not PYBOY_AVAILABLE:
        print("PyBoy not available - skipping Pokemon example")
        return

    print("\n" + "="*60)
    print("POKEMON GEN 1 RL EXAMPLE")
    print("="*60)

    rom_path = "path/to/pokemon_red.gb"
    if not os.path.exists(rom_path):
        print(f"ROM not found at {rom_path}")
        print("This is a demonstration - actual ROM file needed")
        return

    try:
        pyboy = PyBoy(
            rom_path,
            window_type="null",
            game_wrapper=True
        )

        wrapper = pyboy.game_wrapper
        if not isinstance(wrapper, RLGameWrapperPokemonGen1):
            print("RL wrapper not available - using standard wrapper")
            return

        print(f"Starting Pokemon training...")
        print(f"Action space: {wrapper.get_action_names()}")

        # Pokemon training is much longer, so we'll do a short demo
        num_episodes = 20

        for episode in range(num_episodes):
            state = wrapper.reset()
            total_reward = 0

            # Shorter episodes for demo
            for step in range(5000):
                action = np.random.randint(0, wrapper.get_action_space_size())
                next_state = wrapper.step(action)
                total_reward += next_state.reward
                state = next_state

                if next_state.done:
                    break

            # Get Pokemon-specific progress
            pokedex = wrapper.get_pokedex_progress()
            gym = wrapper.get_gym_progress()
            exploration = wrapper.get_exploration_progress()

            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
                  f"Pokedex={pokedex['pokemon_caught']}/151, "
                  f"Badges={gym['badges_earned']}/8, "
                  f"Areas={exploration['unique_maps_visited']}")

        pyboy.stop()

    except Exception as e:
        print(f"Error in Pokemon example: {e}")


def example_general_game():
    """Example using general Game Boy RL wrapper."""
    if not PYBOY_AVAILABLE:
        print("PyBoy not available - skipping general game example")
        return

    print("\n" + "="*60)
    print("GENERAL GAME BOY RL EXAMPLE")
    print("="*60)

    # This example works with any Game Boy ROM
    rom_path = "path/to/any_game.gb"
    if not os.path.exists(rom_path):
        print(f"ROM not found at {rom_path}")
        print("This is a demonstration - actual ROM file needed")
        return

    try:
        pyboy = PyBoy(
            rom_path,
            window_type="null",
            game_wrapper=True
        )

        wrapper = pyboy.game_wrapper
        if not isinstance(wrapper, RLGameWrapperGeneral):
            print("General RL wrapper not available")
            return

        print(f"Starting general game training...")
        print(f"Detected game type: {wrapper.game_type}")

        # Game analysis
        analysis = wrapper.get_game_analysis()
        print(f"Game analysis: {analysis}")

        num_episodes = 30

        for episode in range(num_episodes):
            state = wrapper.reset()
            total_reward = 0

            for step in range(3000):
                action = np.random.randint(0, wrapper.get_action_space_size())
                next_state = wrapper.step(action)
                total_reward += next_state.reward
                state = next_state

                if next_state.done:
                    break

            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
                  f"Steps={step + 1}, Deaths={wrapper.death_count}")

        pyboy.stop()

    except Exception as e:
        print(f"Error in general game example: {e}")


def example_visualization_demo():
    """Demonstrate visualization capabilities without requiring ROMs."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMO")
    print("="*60)

    try:
        from pyboy.plugins.rl_visualization import demo_visualization
        print("Running visualization demo...")
        demo_visualization()
        print("Visualization demo completed!")

    except Exception as e:
        print(f"Error in visualization demo: {e}")


def main():
    """Main function to run all examples."""
    print("PyBoy RL Game Wrapper Examples")
    print("="*60)
    print("This script demonstrates the enhanced RL game wrappers for PyBoy.")
    print("Note: Actual ROM files are required for the game-specific examples.")
    print("")

    # Check if PyBoy is available
    if not PYBOY_AVAILABLE:
        print("PyBoy is not available. Please install PyBoy to run the examples.")
        # Still run visualization demo
        example_visualization_demo()
        return

    # Run examples
    examples = [
        ("Super Mario Land", example_super_mario_land),
        ("Tetris", example_tetris),
        ("Pokemon Gen 1", example_pokemon),
        ("General Game", example_general_game),
        ("Visualization Demo", example_visualization_demo),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples):
        print(f"{i + 1}. {name}")

    try:
        choice = input("\nEnter example number (or 'all' to run all): ").strip().lower()

        if choice == 'all':
            for name, example_func in examples:
                print(f"\nRunning {name} example...")
                example_func()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, example_func = examples[idx]
                print(f"\nRunning {name} example...")
                example_func()
            else:
                print("Invalid choice")
        else:
            print("Invalid input")

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")

    print("\nExamples completed!")


if __name__ == "__main__":
    main()