"""
Quick visualization script for analyzing experiment results
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, obstacle_center, obstacle_radius, save_path=None):
    """
    Visualize robot end-effector trajectory with obstacle.
    
    Parameters
    ----------
    states : ndarray, shape (T, 4)
        State trajectory [q1, q2, dq1, dq2]
    obstacle_center : ndarray, shape (2,)
        Obstacle center position [x, y]
    obstacle_radius : float
        Obstacle safety radius
    save_path : str, optional
        Path to save figure
    """
    from crphcbf.environments import TwoLinkRobotEnv
    
    env = TwoLinkRobotEnv()
    
    # Compute end-effector positions
    ee_positions = np.array([env.forward_kinematics(s[:2]) for s in states])
    
    plt.figure(figsize=(8, 8))
    plt.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=10, label='Start')
    plt.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'r*', markersize=15, label='End')
    
    # Draw obstacle
    circle = plt.Circle(obstacle_center, obstacle_radius, color='r', alpha=0.3, label='Obstacle')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('End-Effector Trajectory', fontsize=14)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_results_comparison(results, save_path=None):
    """
    Create bar chart comparing method performance.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_evaluation()
    save_path : str, optional
        Path to save figure
    """
    methods = list(results.keys())
    success_rates = [results[m]['success_rate'] for m in methods]
    violations = [results[m]['violations_per_episode'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    ax1.bar(methods, success_rates, color=['gray', 'orange', 'blue', 'green'])
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Task Success Rate', fontsize=14)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Violations
    ax2.bar(methods, violations, color=['gray', 'orange', 'blue', 'green'])
    ax2.set_ylabel('Violations per Episode', fontsize=12)
    ax2.set_title('Safety Violations', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    from experiments.run_benchmark import run_evaluation
    
    print("Running benchmark evaluation...")
    results = run_evaluation(num_episodes=20, uncertainty="high", verbose=True)
    
    print("Generating comparison plot...")
    plot_results_comparison(results, save_path='results_comparison.png')
    print("Saved to: results_comparison.png")
