"""
Benchmark Evaluation Harness
============================
Rigorous comparison of four safety-critical control methods:
  1. Unconstrained Control (Baseline)
  2. Nominal CBF-QP
  3. Conservative Robust CBF
  4. CR-PH-CBF (Proposed)
"""

import numpy as np
import time

from crphcbf.environments import TwoLinkRobotEnv
from crphcbf.models import PHResNet
from crphcbf.controllers import SafeCBFFilter
from crphcbf.utils import ExperimentConfig, train_ph_model


def run_single_episode(env, cbf_filter, mode, config):
    """
    Execute one evaluation episode.
    
    Parameters
    ----------
    env : TwoLinkRobotEnv
        Robot environment
    cbf_filter : SafeCBFFilter
        Safety filter instance
    mode : str
        Control mode to evaluate
    config : ExperimentConfig
        Configuration parameters
        
    Returns
    -------
    metrics : dict
        Episode metrics (success, violations, penetration, deadlocks, reward)
    """
    state = env.reset()
    total_reward = 0.0
    violations = 0
    max_penetration = 0.0
    deadlock_steps = 0
    
    for step in range(config.MAX_STEPS):
        q = state[:2]
        dq = state[2:]
        
        # Nominal PD control directed toward goal
        u_nom = config.PD_GAINS['kp'] * (env.goal_q - q) - config.PD_GAINS['kd'] * dq
        
        # Apply CBF Safety Filter
        if mode == "unconstrained":
            u_safe = np.clip(u_nom, config.TORQUE_LIMITS[0], config.TORQUE_LIMITS[1])
            feasible = True
        else:
            u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode=mode)
            
        if not feasible:
            deadlock_steps += 1
        
        # Step physics simulation
        next_state, reward, h_val, ddq_true = env.step(u_safe)
        total_reward += reward
        
        # Check barrier penetration (h < 0 means collision)
        if h_val < 0:
            violations += 1
            max_penetration = max(max_penetration, -h_val)
            
        # Compute residual prediction error & update conformal tracker
        import torch
        with torch.no_grad():
            pred_res = cbf_filter.model(
                torch.FloatTensor(q).unsqueeze(0),
                torch.FloatTensor(dq).unsqueeze(0)
            ).numpy().flatten()
        res_err = np.linalg.norm(ddq_true - pred_res)
        cbf_filter.update_conformal_margin(res_err)
        
        state = next_state
        
    # Success criterion: reach goal without violations
    dist_final = np.linalg.norm(state[:2] - env.goal_q)
    success = 1.0 if (dist_final < 0.2 and violations == 0) else 0.0
    
    return {
        'success': success,
        'violations': violations,
        'max_penetration': max_penetration,
        'deadlock_rate': (deadlock_steps / config.MAX_STEPS) * 100.0,
        'reward': total_reward
    }


def evaluate_method(mode, ph_model, env, config):
    """
    Evaluate one control method over multiple episodes.
    
    Parameters
    ----------
    mode : str
        Control mode to evaluate
    ph_model : PHResNet
        Pre-trained Port-Hamiltonian model
    env : TwoLinkRobotEnv
        Robot environment
    config : ExperimentConfig
        Configuration parameters
        
    Returns
    -------
    results : dict
        Aggregated statistics across all episodes
    """
    cbf_filter = SafeCBFFilter(
        env, ph_model,
        k1=config.CBF_GAINS['k1'],
        k2=config.CBF_GAINS['k2'],
        delta=config.CONFORMAL_DELTA
    )
    
    episode_metrics = []
    
    for ep in range(config.NUM_EPISODES):
        metrics = run_single_episode(env, cbf_filter, mode, config)
        episode_metrics.append(metrics)
    
    # Aggregate statistics
    results = {
        'success_rate': np.mean([m['success'] for m in episode_metrics]) * 100.0,
        'violations_per_episode': np.mean([m['violations'] for m in episode_metrics]),
        'max_violation': np.mean([m['max_penetration'] for m in episode_metrics]),
        'deadlock_rate': np.mean([m['deadlock_rate'] for m in episode_metrics]),
        'mean_reward': np.mean([m['reward'] for m in episode_metrics]),
        'std_reward': np.std([m['reward'] for m in episode_metrics])
    }
    
    return results


def run_evaluation(num_episodes=20, uncertainty="high", verbose=True):
    """
    Run complete benchmark evaluation comparing all methods.
    
    Parameters
    ----------
    num_episodes : int, optional
        Number of episodes per method (default: 20)
    uncertainty : str, optional
        Dynamics uncertainty level: "low", "medium", or "high" (default: "high")
    verbose : bool, optional
        Print detailed progress (default: True)
        
    Returns
    -------
    results : dict
        Results for all evaluated methods
    """
    # Set configuration
    config = ExperimentConfig()
    config.NUM_EPISODES = num_episodes
    config.UNCERTAINTY_LEVEL = uncertainty
    config.set_seed()
    
    if verbose:
        print("="*75)
        print(f"CR-PH-CBF BENCHMARK EVALUATION")
        print(f"Uncertainty Level: {uncertainty.upper()}")
        print(f"Episodes per method: {num_episodes}")
        print("="*75 + "\n")
    
    # Initialize environment and model
    env = TwoLinkRobotEnv(uncertainty_level=uncertainty)
    ph_model = PHResNet(hidden_dim=config.HIDDEN_DIM)
    
    # Pre-train Port-Hamiltonian model
    train_ph_model(
        ph_model, env,
        num_iterations=config.TRAINING_ITERATIONS,
        lr=config.LEARNING_RATE,
        verbose=verbose
    )
    
    # Evaluate all methods
    results = {}
    t_start = time.time()
    
    for mode in config.MODES:
        if verbose:
            print(f"Evaluating: {mode:<25}", end=" ... ", flush=True)
        
        mode_results = evaluate_method(mode, ph_model, env, config)
        results[mode] = mode_results
        
        if verbose:
            print(f"Done ({time.time() - t_start:.1f}s elapsed)")
    
    # Print results table
    if verbose:
        print_results_table(results, config.MODES)
    
    return results


def print_results_table(results, modes):
    """Print publication-ready results table."""
    print("\n" + "="*95)
    print("EMPIRICAL RESULTS (Reproducible Data for Publication)")
    print("="*95)
    print(f"{'Method':<25} | {'Success (%)':<12} | {'Violations':<11} | "
          f"{'Max Pen (m²)':<13} | {'Deadlock (%)':<12} | {'Reward':<15}")
    print("-" * 95)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<25} | {r['success_rate']:>11.1f}% | "
              f"{r['violations_per_episode']:>10.1f}  | "
              f"{r['max_violation']:>12.3f}  | "
              f"{r['deadlock_rate']:>11.1f}% | "
              f"{r['mean_reward']:>6.1f} ± {r['std_reward']:<5.1f}")
    
    print("="*95 + "\n")


if __name__ == "__main__":
    results = run_evaluation(num_episodes=20, uncertainty="high")
