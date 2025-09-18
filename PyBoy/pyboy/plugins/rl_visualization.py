#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
"""
RL Visualization and Performance Metrics Tools

This module provides comprehensive visualization and performance analysis tools
for reinforcement learning agents playing Game Boy games.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import numpy as np
import pyboy
from pyboy.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics data class."""
    episode: int
    step: int
    reward: float
    length: int
    survival_time: float
    final_score: float
    success: bool
    game_specific_metrics: Dict[str, Any]
    timestamp: float
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None


class RLVisualizer:
    """
    Real-time visualization and performance analysis for RL agents.

    Features:
    - Live plotting of training metrics
    - Performance dashboards
    - Agent behavior analysis
    - Export capabilities
    - Real-time monitoring
    """

    def __init__(self, output_dir: str = "rl_visualizations", auto_save: bool = True):
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.enabled = VISUALIZATION_AVAILABLE

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Metrics storage
        self.metrics_history = deque(maxlen=10000)
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=1000)
        self.running_rewards = deque(maxlen=100)

        # Real-time data
        self.current_episode = 0
        self.current_step = 0
        self.current_reward = 0.0
        self.start_time = time.time()

        # Visualization settings
        self.update_interval = 10  # Update every 10 episodes
        self.plot_styles = {
            'reward': 'blue',
            'length': 'green',
            'success': 'orange',
            'moving_avg': 'red'
        }

        # Thread safety
        self.lock = threading.Lock()
        self._setup_plots()

    def _setup_plots(self):
        """Initialize matplotlib figures."""
        if not self.enabled:
            return

        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('RL Training Dashboard', fontsize=16, fontweight='bold')

        # Initialize subplot titles
        subplot_titles = [
            'Episode Rewards', 'Episode Lengths', 'Success Rate',
            'Reward Distribution', 'Learning Progress', 'Action Analysis'
        ]

        for ax, title in zip(self.axes.flat, subplot_titles):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics to the visualization."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.episode_rewards.append(metrics.reward)
            self.episode_lengths.append(metrics.length)
            self.success_rates.append(float(metrics.success))

            # Calculate running average
            if len(self.episode_rewards) >= 10:
                running_avg = np.mean(list(self.episode_rewards)[-10:])
                self.running_rewards.append(running_avg)

            self.current_episode = metrics.episode
            self.current_step = metrics.step

        # Update plots if interval reached
        if metrics.episode % self.update_interval == 0:
            self.update_plots()

    def update_plots(self):
        """Update all visualization plots."""
        if not self.enabled:
            return

        try:
            with self.lock:
                if len(self.metrics_history) == 0:
                    return

                self._plot_rewards()
                self._plot_lengths()
                self._plot_success_rate()
                self._plot_reward_distribution()
                self._plot_learning_progress()
                self._plot_action_analysis()

                if self.auto_save:
                    self.save_current_plots()

        except Exception as e:
            logger.warning(f"Plot update failed: {e}")

    def _plot_rewards(self):
        """Plot episode rewards over time."""
        ax = self.axes[0, 0]
        ax.clear()

        episodes = range(1, len(self.episode_rewards) + 1)
        ax.plot(episodes, self.episode_rewards, color=self.plot_styles['reward'],
                alpha=0.6, linewidth=1, label='Episode Reward')

        # Plot moving average
        if len(self.running_rewards) > 0:
            window_start = len(self.episode_rewards) - len(self.running_rewards)
            ax.plot(range(window_start + 1, len(self.episode_rewards) + 1),
                   self.running_rewards, color=self.plot_styles['moving_avg'],
                   linewidth=2, label=f'{len(self.running_rewards[0])}-episode MA')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_lengths(self):
        """Plot episode lengths over time."""
        ax = self.axes[0, 1]
        ax.clear()

        episodes = range(1, len(self.episode_lengths) + 1)
        ax.plot(episodes, self.episode_lengths, color=self.plot_styles['length'],
                alpha=0.6, linewidth=1)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)

    def _plot_success_rate(self):
        """Plot rolling success rate."""
        ax = self.axes[0, 2]
        ax.clear()

        if len(self.success_rates) > 10:
            # Calculate rolling success rate
            window_size = min(50, len(self.success_rates))
            rolling_success = []
            for i in range(window_size, len(self.success_rates) + 1):
                success_rate = np.mean(list(self.success_rates)[i-window_size:i])
                rolling_success.append(success_rate)

            episodes = range(window_size, len(self.success_rates) + 1)
            ax.plot(episodes, rolling_success, color=self.plot_styles['success'],
                    linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (Rolling Window)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_reward_distribution(self):
        """Plot reward distribution histogram."""
        ax = self.axes[1, 0]
        ax.clear()

        if len(self.episode_rewards) > 0:
            ax.hist(self.episode_rewards, bins=30, alpha=0.7, color=self.plot_styles['reward'],
                   edgecolor='white', linewidth=0.5)
            ax.axvline(np.mean(self.episode_rewards), color='red', linestyle='--',
                      label=f'Mean: {np.mean(self.episode_rewards):.2f}')

        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_learning_progress(self):
        """Plot learning progress over time."""
        ax = self.axes[1, 1]
        ax.clear()

        if len(self.metrics_history) > 0:
            # Extract learning rates and epsilon values
            episodes = []
            learning_rates = []
            epsilons = []

            for metrics in self.metrics_history:
                if metrics.learning_rate is not None:
                    episodes.append(metrics.episode)
                    learning_rates.append(metrics.learning_rate)
                if metrics.epsilon is not None:
                    epsilons.append(metrics.epsilon)

            # Plot learning rates
            if learning_rates:
                ax2 = ax.twinx()
                ax2.plot(episodes, learning_rates, color='cyan', linewidth=2,
                        label='Learning Rate', alpha=0.8)
                ax2.set_ylabel('Learning Rate', color='cyan')
                ax2.tick_params(axis='y', labelcolor='cyan')

            # Plot epsilon decay
            if epsilons:
                ax.plot(episodes, epsilons, color='yellow', linewidth=2,
                       label='Epsilon', alpha=0.8)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Learning Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_action_analysis(self):
        """Plot action distribution and analysis."""
        ax = self.axes[1, 2]
        ax.clear()

        # This would be populated with actual action data
        # For now, show placeholder
        actions = ['Up', 'Down', 'Left', 'Right', 'A', 'B', 'Start', 'Select']
        action_counts = np.random.randint(10, 100, len(actions))  # Placeholder

        ax.bar(actions, action_counts, alpha=0.7, color='lightblue',
               edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    def save_current_plots(self, filename: Optional[str] = None):
        """Save current plots to file."""
        if not self.enabled:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_dashboard_{timestamp}.png"

        filepath = os.path.join(self.output_dir, filename)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training dashboard to {filepath}")

    def create_interactive_dashboard(self) -> Optional[str]:
        """Create interactive Plotly dashboard."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return None

        try:
            with self.lock:
                if len(self.metrics_history) == 0:
                    return None

                # Create subplots
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=[
                        'Episode Rewards', 'Episode Lengths',
                        'Success Rate', 'Reward Distribution',
                        'Training Progress', 'Performance Summary'
                    ],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": True}, {"type": "table"}]]
                )

                episodes = list(range(1, len(self.episode_rewards) + 1))

                # Episode rewards
                fig.add_trace(
                    go.Scatter(x=episodes, y=list(self.episode_rewards),
                              mode='lines', name='Episode Reward',
                              line=dict(color='blue', width=1)),
                    row=1, col=1
                )

                # Episode lengths
                fig.add_trace(
                    go.Scatter(x=episodes, y=list(self.episode_lengths),
                              mode='lines', name='Episode Length',
                              line=dict(color='green', width=1)),
                    row=1, col=2
                )

                # Success rate
                if len(self.success_rates) > 10:
                    window_size = min(50, len(self.success_rates))
                    rolling_success = []
                    for i in range(window_size, len(self.success_rates) + 1):
                        success_rate = np.mean(list(self.success_rates)[i-window_size:i])
                        rolling_success.append(success_rate)

                    fig.add_trace(
                        go.Scatter(x=list(range(window_size, len(self.success_rates) + 1)),
                                  y=rolling_success,
                                  mode='lines', name='Success Rate',
                                  line=dict(color='orange', width=2)),
                        row=2, col=1
                    )

                # Reward distribution
                fig.add_trace(
                    go.Histogram(x=list(self.episode_rewards), nbinsx=30,
                                name='Reward Distribution',
                                marker_color='blue', opacity=0.7),
                    row=2, col=2
                )

                # Training progress
                if len(self.metrics_history) > 0:
                    timestamps = [m.timestamp - self.start_time for m in self.metrics_history]
                    cumulative_rewards = []
                    total_reward = 0

                    for m in self.metrics_history:
                        total_reward += m.reward
                        cumulative_rewards.append(total_reward)

                    fig.add_trace(
                        go.Scatter(x=timestamps, y=cumulative_rewards,
                                  mode='lines', name='Cumulative Reward',
                                  line=dict(color='red', width=2)),
                        row=3, col=1
                    )

                # Performance summary table
                summary_data = {
                    'Metric': ['Total Episodes', 'Avg Reward', 'Avg Length', 'Success Rate', 'Best Reward'],
                    'Value': [
                        len(self.episode_rewards),
                        f"{np.mean(self.episode_rewards):.2f}",
                        f"{np.mean(self.episode_lengths):.1f}",
                        f"{np.mean(self.success_rates):.2%}",
                        f"{max(self.episode_rewards):.2f}"
                    ]
                }

                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value'],
                                  fill_color='lightblue'),
                        cells=dict(values=[summary_data['Metric'], summary_data['Value']],
                                  fill_color='white')
                    ),
                    row=3, col=2
                )

                fig.update_layout(
                    title_text="RL Training Interactive Dashboard",
                    showlegend=False,
                    height=800
                )

                # Save interactive dashboard
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"interactive_dashboard_{timestamp}.html"
                filepath = os.path.join(self.output_dir, filename)
                fig.write_html(filepath)

                logger.info(f"Saved interactive dashboard to {filepath}")
                return filepath

        except Exception as e:
            logger.warning(f"Interactive dashboard creation failed: {e}")
            return None

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        with self.lock:
            if len(self.metrics_history) == 0:
                return "No metrics data available for report."

            report = []
            report.append("=" * 60)
            report.append("RL TRAINING PERFORMANCE REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Total Episodes: {len(self.metrics_history)}")
            report.append(f"Total Steps: {sum(m.length for m in self.metrics_history)}")
            report.append(f"Training Duration: {time.time() - self.start_time:.1f} seconds")
            report.append("")

            # Reward analysis
            rewards = [m.reward for m in self.metrics_history]
            report.append("REWARD ANALYSIS:")
            report.append(f"  Mean Reward: {np.mean(rewards):.2f}")
            report.append(f"  Median Reward: {np.median(rewards):.2f}")
            report.append(f"  Std Reward: {np.std(rewards):.2f}")
            report.append(f"  Min Reward: {min(rewards):.2f}")
            report.append(f"  Max Reward: {max(rewards):.2f}")
            report.append(f"  95th Percentile: {np.percentile(rewards, 95):.2f}")
            report.append("")

            # Episode length analysis
            lengths = [m.length for m in self.metrics_history]
            report.append("EPISODE LENGTH ANALYSIS:")
            report.append(f"  Mean Length: {np.mean(lengths):.1f} steps")
            report.append(f"  Median Length: {np.median(lengths):.1f} steps")
            report.append(f"  Min Length: {min(lengths)} steps")
            report.append(f"  Max Length: {max(lengths)} steps")
            report.append("")

            # Success rate analysis
            success_count = sum(1 for m in self.metrics_history if m.success)
            success_rate = success_count / len(self.metrics_history)
            report.append("SUCCESS ANALYSIS:")
            report.append(f"  Success Rate: {success_rate:.2%}")
            report.append(f"  Successful Episodes: {success_count}")
            report.append(f"  Failed Episodes: {len(self.metrics_history) - success_count}")
            report.append("")

            # Recent performance
            if len(self.metrics_history) >= 10:
                recent = list(self.metrics_history)[-10:]
                recent_rewards = [m.reward for m in recent]
                recent_success = sum(1 for m in recent if m.success)

                report.append("RECENT PERFORMANCE (Last 10 episodes):")
                report.append(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                report.append(f"  Success Rate: {recent_success/len(recent):.2%}")
                report.append("")

            # Learning progress
            if len(self.metrics_history) >= 50:
                early = list(self.metrics_history)[:10]
                late = list(self.metrics_history)[-10:]

                early_avg = np.mean([m.reward for m in early])
                late_avg = np.mean([m.reward for m in late])
                improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0

                report.append("LEARNING PROGRESS:")
                report.append(f"  Early Avg Reward (first 10): {early_avg:.2f}")
                report.append(f"  Late Avg Reward (last 10): {late_avg:.2f}")
                report.append(f"  Improvement: {improvement:+.1f}%")
                report.append("")

            # Recommendations
            report.append("RECOMMENDATIONS:")
            if success_rate < 0.1:
                report.append("  - Consider adjusting reward function")
                report.append("  - Increase exploration in early training")
            elif success_rate > 0.8:
                report.append("  - Consider increasing task difficulty")
                report.append("  - Focus on optimizing performance")

            if np.std(rewards) > np.mean(rewards):
                report.append("  - High reward variance detected")
                report.append("  - Consider reward normalization")

            report.append("")

        return "\n".join(report)

    def export_metrics(self, filename: Optional[str] = None):
        """Export all metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        with self.lock:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_episodes': len(self.metrics_history),
                    'total_steps': sum(m.length for m in self.metrics_history),
                    'training_duration': time.time() - self.start_time,
                },
                'metrics': [asdict(m) for m in self.metrics_history],
                'summary': {
                    'avg_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
                    'avg_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
                    'success_rate': float(np.mean(self.success_rates)) if self.success_rates else 0,
                    'best_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0,
                }
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported metrics to {filepath}")
        return filepath


class RealTimeMonitor:
    """Real-time monitoring system for RL training."""

    def __init__(self, port: int = 8080):
        self.port = port
        self.running = False
        self.metrics_buffer = deque(maxlen=100)
        self.start_time = time.time()

    def add_metric(self, episode: int, reward: float, length: int, **kwargs):
        """Add real-time metric."""
        metric = {
            'timestamp': time.time(),
            'episode': episode,
            'reward': reward,
            'length': length,
            'elapsed_time': time.time() - self.start_time,
            **kwargs
        }
        self.metrics_buffer.append(metric)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        if len(self.metrics_buffer) == 0:
            return {}

        recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 episodes

        return {
            'total_episodes': len(self.metrics_buffer),
            'current_episode': self.metrics_buffer[-1]['episode'],
            'avg_reward_10': np.mean([m['reward'] for m in recent_metrics]),
            'avg_length_10': np.mean([m['length'] for m in recent_metrics]),
            'training_time': time.time() - self.start_time,
            'eps_per_minute': len(self.metrics_buffer) / max((time.time() - self.start_time) / 60, 0.1),
        }

    def start_server(self):
        """Start monitoring server (placeholder)."""
        logger.info(f"Real-time monitoring server started on port {self.port}")
        # In a real implementation, this would start a web server
        self.running = True

    def stop_server(self):
        """Stop monitoring server."""
        self.running = False
        logger.info("Real-time monitoring server stopped")


def create_visualization_suite(output_dir: str = "rl_analysis") -> Dict[str, Any]:
    """Create a complete visualization suite."""
    suite = {
        'visualizer': RLVisualizer(output_dir),
        'monitor': RealTimeMonitor(),
        'output_dir': output_dir,
    }

    logger.info(f"Created RL visualization suite in {output_dir}")
    return suite


# Example usage functions
def demo_visualization():
    """Demonstrate visualization capabilities."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available - install matplotlib")
        return

    viz = RLVisualizer("demo_output", auto_save=False)

    # Generate demo data
    np.random.seed(42)
    for episode in range(100):
        reward = np.random.normal(episode * 0.5, 10)
        length = np.random.randint(100, 1000)
        success = reward > np.random.normal(episode * 0.3, 5)

        metrics = TrainingMetrics(
            episode=episode,
            step=length,
            reward=reward,
            length=length,
            survival_time=length / 60.0,
            final_score=reward * 10,
            success=success,
            game_specific_metrics={'demo': True},
            timestamp=time.time(),
            learning_rate=0.001 * (0.99 ** episode),
            epsilon=1.0 * (0.995 ** episode)
        )

        viz.add_metrics(metrics)

    # Create visualizations
    viz.update_plots()
    viz.save_current_plots("demo_dashboard.png")

    if PLOTLY_AVAILABLE:
        interactive_path = viz.create_interactive_dashboard()
        if interactive_path:
            print(f"Interactive dashboard: {interactive_path}")

    # Generate report
    report = viz.generate_performance_report()
    print(report)

    # Export metrics
    export_path = viz.export_metrics("demo_metrics.json")
    print(f"Metrics exported to: {export_path}")


if __name__ == "__main__":
    demo_visualization()