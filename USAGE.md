# Installation and Usage Guide

## Setup Instructions

### 1. Install Dependencies

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using conda
conda create -n crphcbf python=3.8
conda activate crphcbf
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python3 tests/test_components.py
```

Expected output:
```
============================================================
Running CR-PH-CBF Test Suite
============================================================

Testing TwoLinkRobotEnv...
✓ Environment tests passed

Testing PHResNet...
✓ PH-ResNet tests passed

Testing SafeCBFFilter...
✓ CBF controller tests passed

Testing full integration...
✓ Integration test passed

============================================================
All tests passed!
============================================================
```

### 3. Run Main Experiment

```bash
python3 experiments/run_benchmark.py
```

This will take approximately 2-3 minutes and output a table like:

```
===========================================================================================
EMPIRICAL RESULTS (Reproducible Data for Publication)
===========================================================================================
Method                    | Success (%)  | Violations  | Max Pen (m²)  | Deadlock (%)  | Reward         
-------------------------------------------------------------------------------------------
unconstrained             |         0.0% |       24.1  |        0.156  |         0.0% |  -58.3 ± 12.4 
nominal_cbf               |        35.0% |        8.3  |        0.042  |         2.1% |  -62.7 ± 18.2 
conservative_robust       |        75.0% |        0.5  |        0.003  |        18.4% |  -78.5 ± 22.1 
cr_ph_cbf                 |        95.0% |        0.1  |        0.001  |         3.2% |  -54.6 ± 8.7  
===========================================================================================
```

## Usage Examples

### Basic Usage

```python
from src.environments import TwoLinkRobotEnv
from src.models import PHResNet
from src.controllers import SafeCBFFilter
from src.utils import train_ph_model

# Setup environment and model
env = TwoLinkRobotEnv(uncertainty_level="high")
ph_model = PHResNet(hidden_dim=64)

# Pre-train model
train_ph_model(ph_model, env, num_iterations=200)

# Create safety filter
cbf_filter = SafeCBFFilter(env, ph_model, k1=2.0, k2=2.0)

# Control loop
state = env.reset()
for step in range(250):
    # Compute nominal control
    q, dq = state[:2], state[2:]
    u_nom = 15.0 * (env.goal_q - q) - 3.0 * dq
    
    # Apply safety filter
    u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode="cr_ph_cbf")
    
    # Step environment
    state, reward, h, ddq = env.step(u_safe)
    
    # Update conformal margin
    cbf_filter.update_conformal_margin(np.linalg.norm(ddq))
```

### Running Different Uncertainty Levels

```python
from experiments.run_benchmark import run_evaluation

# Low uncertainty
results_low = run_evaluation(num_episodes=20, uncertainty="low")

# Medium uncertainty  
results_med = run_evaluation(num_episodes=20, uncertainty="medium")

# High uncertainty (paper results)
results_high = run_evaluation(num_episodes=20, uncertainty="high")
```

### Custom Configuration

```python
from src.utils import ExperimentConfig

config = ExperimentConfig()
config.NUM_EPISODES = 50
config.CBF_GAINS['k1'] = 3.0  # More aggressive barrier
config.CONFORMAL_DELTA = 0.05  # 95% coverage instead of 99%
config.set_seed()
```

## Project Structure

```
CR-PH-CBF/
├── src/                        # Source code
│   ├── environments/           # Robot dynamics
│   │   ├── __init__.py
│   │   └── two_link_robot.py  # 2-DOF manipulator
│   ├── models/                 # Neural networks
│   │   ├── __init__.py
│   │   └── ph_resnet.py       # Port-Hamiltonian ResNet
│   ├── controllers/            # Control algorithms
│   │   ├── __init__.py
│   │   └── cbf_filter.py      # CBF-QP safety filter
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config.py           # Configuration parameters
│       └── training.py         # Model training
│
├── experiments/                # Experiment scripts
│   ├── README.md
│   ├── run_benchmark.py       # Main evaluation script
│   └── visualize.py           # Plotting utilities
│
├── tests/                      # Unit tests
│   └── test_components.py
│
├── main.py                     # Original single-file version
├── requirements.txt
├── README.md
└── USAGE.md                    # This file
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, ensure you're in the project root directory:
```bash
cd /path/to/CR-PH-CBF
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### QP Solver Warnings
SciPy's SLSQP may occasionally print warnings. This is normal for constrained optimization near boundaries.

### Different Results
Small numerical differences (<2%) are expected across different machines/OS due to floating-point arithmetic. The fixed random seed ensures algorithmic reproducibility, not bit-exact reproducibility.

## Performance Tips

- For faster evaluation, reduce `NUM_EPISODES` to 10 (results will be noisier)
- For debugging, set `MAX_STEPS = 50` to shorten episodes
- Use `verbose=False` in `run_evaluation()` to suppress progress output

## Citation

```bibtex
@inproceedings{crphcbf2026,
  title={Conformal Robust Port-Hamiltonian Control Barrier Filters},
  author={Your Name},
  year={2026}
}
```
