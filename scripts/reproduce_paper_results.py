#!/usr/bin/env python3
"""
Reproduce Main Paper Results (Table 1)
======================================

This script reproduces the main experimental results reported in Table 1
of the paper. All experiments use fixed random seeds for reproducibility.

Usage:
    python scripts/reproduce_paper_results.py
    python scripts/reproduce_paper_results.py --uncertainty high --episodes 20
    python scripts/reproduce_paper_results.py --save-path results/paper/table1.json

Output:
    - Console: Formatted table matching paper
    - JSON: Numerical results with metadata
    - CSV: Results for plotting
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crphcbf.environments import TwoLinkRobotEnv
from crphcbf.models import PHResNet
from crphcbf.controllers import SafeCBFFilter
from crphcbf.utils import ExperimentConfig, train_ph_model
import numpy as np
import time


def get_git_info():
    """Get current git commit hash and status for reproducibility."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        status = subprocess.check_output(['git', 'status', '--short']).decode('ascii').strip()
        return {
            'commit': commit,
            'dirty': len(status) > 0,
            'status': status if len(status) > 0 else 'clean'
        }
    except:
        return {'commit': 'unknown', 'dirty': False, 'status': 'unknown'}


def run_single_episode(env, cbf_filter, mode, config):
    """Execute one evaluation episode."""
    state = env.reset()
    total_reward = 0.0
    violations = 0
    max_penetration = 0.0
    deadlock_steps = 0
    
    for step in range(config.MAX_STEPS):
        q = state[:2]
        dq = state[2:]
        
        # Nominal PD control
        u_nom = config.PD_GAINS['kp'] * (env.goal_q - q) - config.PD_GAINS['kd'] * dq
        
        # Apply CBF filter
        if mode == "unconstrained":
            u_safe = np.clip(u_nom, config.TORQUE_LIMITS[0], config.TORQUE_LIMITS[1])
            feasible = True
        else:
            u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode=mode)
            
        if not feasible:
            deadlock_steps += 1
        
        # Step environment
        next_state, reward, h_val, ddq_true = env.step(u_safe)
        total_reward += reward
        
        # Check safety violations
        if h_val < 0:
            violations += 1
            max_penetration = max(max_penetration, -h_val)
            
        # Update conformal margin
        import torch
        with torch.no_grad():
            pred_res = cbf_filter.model(
                torch.FloatTensor(q).unsqueeze(0),
                torch.FloatTensor(dq).unsqueeze(0)
            ).numpy().flatten()
        res_err = np.linalg.norm(ddq_true - pred_res)
        cbf_filter.update_conformal_margin(res_err)
        
        state = next_state
    
    # Success criterion
    dist_final = np.linalg.norm(state[:2] - env.goal_q)
    success = 1.0 if (dist_final < 0.2 and violations == 0) else 0.0
    
    return {
        'success': success,
        'violations': violations,
        'max_penetration': max_penetration,
        'deadlock_rate': (deadlock_steps / config.MAX_STEPS) * 100.0,
        'reward': total_reward
    }


def evaluate_method(mode, ph_model, env, config, verbose=True):
    """Evaluate one control method."""
    cbf_filter = SafeCBFFilter(
        env, ph_model,
        k1=config.CBF_GAINS['k1'],
        k2=config.CBF_GAINS['k2'],
        delta=config.CONFORMAL_DELTA
    )
    
    episode_metrics = []
    
    if verbose:
        print(f"  Running {config.NUM_EPISODES} episodes for {mode}...", end=' ', flush=True)
    
    for ep in range(config.NUM_EPISODES):
        metrics = run_single_episode(env, cbf_filter, mode, config)
        episode_metrics.append(metrics)
    
    if verbose:
        print("✓")
    
    # Aggregate statistics
    results = {
        'success_rate': float(np.mean([m['success'] for m in episode_metrics]) * 100.0),
        'violations_per_episode': float(np.mean([m['violations'] for m in episode_metrics])),
        'max_violation': float(np.mean([m['max_penetration'] for m in episode_metrics])),
        'deadlock_rate': float(np.mean([m['deadlock_rate'] for m in episode_metrics])),
        'mean_reward': float(np.mean([m['reward'] for m in episode_metrics])),
        'std_reward': float(np.std([m['reward'] for m in episode_metrics])),
        'episode_data': episode_metrics  # Raw episode data
    }
    
    return results


def print_results_table(results, modes):
    """Print publication-ready table."""
    print("\n" + "="*107)
    print("                          Table 1: Main Results (High Uncertainty)")
    print("="*107)
    print(f"{'Method':<25} | {'Success Rate ↑':<15} | {'Violations ↓':<14} | "
          f"{'Max Pen ↓':<13} | {'Deadlock ↓':<15} | {'Reward':<15}")
    print("-" * 107)
    
    for mode in modes:
        r = results[mode]
        method_name = {
            'unconstrained': 'Unconstrained',
            'nominal_cbf': 'Nominal CBF',
            'conservative_robust': 'Conservative Robust',
            'cr_ph_cbf': 'CR-PH-CBF (Ours)'
        }.get(mode, mode)
        
        print(f"{method_name:<25} | {r['success_rate']:>13.1f}% | "
              f"{r['violations_per_episode']:>13.1f}  | "
              f"{r['max_violation']:>12.3f}  | "
              f"{r['deadlock_rate']:>13.1f}% | "
              f"{r['mean_reward']:>6.1f} ± {r['std_reward']:<5.1f}")
    
    print("="*107 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--uncertainty', type=str, default='high',
                       choices=['low', 'medium', 'high'],
                       help='Dynamics uncertainty level')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes per method')
    parser.add_argument('--save-path', type=str, default='results/paper/table1.json',
                       help='Path to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print("="*107)
    print("                    Reproducing Paper Results (Table 1)")
    print("="*107)
    print(f"Configuration:")
    print(f"  - Uncertainty Level: {args.uncertainty}")
    print(f"  - Episodes per method: {args.episodes}")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Save path: {args.save_path}")
    
    # Get git info for reproducibility
    git_info = get_git_info()
    print(f"  - Git commit: {git_info['commit'][:8]}")
    if git_info['dirty']:
        print(f"    WARNING: Repository has uncommitted changes!")
    print()
    
    # Setup configuration
    config = ExperimentConfig()
    config.NUM_EPISODES = args.episodes
    config.UNCERTAINTY_LEVEL = args.uncertainty
    config.SEED = args.seed
    config.set_seed()
    
    # Initialize environment and model
    print("Initializing environment and model...")
    env = TwoLinkRobotEnv(uncertainty_level=args.uncertainty)
    ph_model = PHResNet(hidden_dim=config.HIDDEN_DIM)
    
    # Pre-train model
    print(f"Pre-training Port-Hamiltonian model ({config.TRAINING_ITERATIONS} iterations)...")
    train_ph_model(
        ph_model, env,
        num_iterations=config.TRAINING_ITERATIONS,
        lr=config.LEARNING_RATE,
        verbose=False
    )
    print("✓ Model training complete\n")
    
    # Evaluate all methods
    print("Running experiments...")
    results = {}
    t_start = time.time()
    
    for mode in config.MODES:
        mode_results = evaluate_method(mode, ph_model, env, config, verbose=True)
        results[mode] = mode_results
    
    elapsed = time.time() - t_start
    print(f"\n✓ All experiments complete ({elapsed:.1f}s)\n")
    
    # Print results
    print_results_table(results, config.MODES)
    
    # Save results
    output_dir = Path(args.save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON with metadata
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'git': git_info,
            'config': {
                'uncertainty': args.uncertainty,
                'episodes': args.episodes,
                'seed': args.seed,
                'max_steps': config.MAX_STEPS,
                'cbf_gains': config.CBF_GAINS,
                'conformal_delta': config.CONFORMAL_DELTA
            }
        },
        'results': results
    }
    
    with open(args.save_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {args.save_path}")
    
    # Also save as CSV for easy plotting
    csv_path = str(args.save_path).replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write("Method,Success_Rate,Violations,Max_Penetration,Deadlock_Rate,Mean_Reward,Std_Reward\n")
        for mode in config.MODES:
            r = results[mode]
            f.write(f"{mode},{r['success_rate']},{r['violations_per_episode']},"
                   f"{r['max_violation']},{r['deadlock_rate']},"
                   f"{r['mean_reward']},{r['std_reward']}\n")
    
    print(f"✓ CSV saved to: {csv_path}")
    print("\nReproduction complete! 🎉")


if __name__ == "__main__":
    main()
