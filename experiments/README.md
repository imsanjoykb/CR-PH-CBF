# Experiment Results

This directory contains experimental results and analysis scripts.

## Running Experiments

### Main Benchmark
```bash
python run_benchmark.py
```

### Visualization
```bash
python visualize.py
```

## Expected Results

The benchmark evaluates four methods:

1. **Unconstrained**: Baseline without safety constraints
2. **Nominal CBF**: Standard CBF assuming perfect model knowledge
3. **Conservative Robust**: Fixed worst-case Lipschitz margin
4. **CR-PH-CBF**: Proposed method with adaptive conformal calibration

### Metrics

- **Success Rate (%)**: Percentage of episodes reaching goal without collisions
- **Violations**: Average number of barrier violations per episode
- **Max Penetration**: Average maximum penetration depth (m²)
- **Deadlock Rate (%)**: Percentage of timesteps where QP is infeasible
- **Reward**: Task performance metric

## Reproducing Paper Results

All experiments use SEED=42 for reproducibility. The results should match Table I in the paper within statistical variance (±2% for success rate).
