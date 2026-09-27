# CR-PH-CBF: Conformal Robust Port-Hamiltonian Control Barrier Filter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation** of the CR-PH-CBF framework for safety-critical robot control under dynamics uncertainty.

## Overview

This repository contains a complete research codebase for empirical verification of the **Conformal Robust Port-Hamiltonian Control Barrier Filter (CR-PH-CBF)** method. The framework combines:

1. **Port-Hamiltonian Neural Networks** (PH-ResNet) for physics-informed dynamics learning
2. **Control Barrier Functions** (CBF) for formal safety guarantees
3. **Adaptive Conformal Prediction** for uncertainty quantification

We benchmark against three baseline methods on a 2-DOF planar manipulator task with obstacle avoidance under severe dynamics uncertainty (payload mismatch, friction stiction, external forces).

### Key Features

- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Reproducible results** (fixed random seeds, deterministic evaluation)
- ✅ **Physics-informed learning** (energy-preserving neural architecture)
- ✅ **Formal safety** (High-Order CBF with conformal calibration)
- ✅ **Publication-ready** (metrics match paper Table I)

---

## Project Structure

```
CR-PH-CBF/
├── src/
│   ├── environments/       # Robot dynamics simulation
│   │   └── two_link_robot.py
│   ├── models/             # Neural network architectures
│   │   └── ph_resnet.py
│   ├── controllers/        # Safety filters and controllers
│   │   └── cbf_filter.py
│   └── utils/              # Configuration and training utilities
│       ├── config.py
│       └── training.py
├── experiments/
│   └── run_benchmark.py    # Main evaluation script
├── main.py                 # Legacy single-file implementation
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CR-PH-CBF.git
cd CR-PH-CBF

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

Run the complete benchmark evaluation:

```bash
python experiments/run_benchmark.py
```

This will:
1. Initialize a 2-DOF robot with high dynamics uncertainty
2. Pre-train the Port-Hamiltonian residual model
3. Evaluate 4 methods over 20 episodes each:
   - **Unconstrained** (no safety filter)
   - **Nominal CBF-QP** (standard CBF)
   - **Conservative Robust CBF** (fixed worst-case margin)
   - **CR-PH-CBF** (proposed method)
4. Print a publication-ready results table

**Expected output:**
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

---

## Usage

### Running with Custom Parameters

```python
from experiments.run_benchmark import run_evaluation

# High uncertainty (default)
results = run_evaluation(num_episodes=20, uncertainty="high")

# Medium uncertainty
results = run_evaluation(num_episodes=50, uncertainty="medium")

# Low uncertainty
results = run_evaluation(num_episodes=100, uncertainty="low")
```

### Using Individual Components

```python
from src.environments import TwoLinkRobotEnv
from src.models import PHResNet
from src.controllers import SafeCBFFilter
from src.utils import ExperimentConfig, train_ph_model

# Setup
config = ExperimentConfig()
config.set_seed()

env = TwoLinkRobotEnv(uncertainty_level="high")
ph_model = PHResNet(hidden_dim=64)

# Train model
train_ph_model(ph_model, env, num_iterations=200)

# Create safety filter
cbf_filter = SafeCBFFilter(env, ph_model, k1=2.0, k2=2.0)

# Run control loop
state = env.reset()
u_nom = compute_nominal_control(state)  # Your nominal controller
u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode="cr_ph_cbf")
next_state, reward, h, ddq = env.step(u_safe)
```

---

## Method Comparison

| Method | Safety Guarantee | Uncertainty Handling | Performance |
|--------|-----------------|---------------------|-------------|
| **Unconstrained** | ❌ None | ❌ None | High task performance, frequent collisions |
| **Nominal CBF** | ⚠️ Assumes perfect model | ❌ None | Moderate violations under uncertainty |
| **Conservative Robust** | ✅ Worst-case Lipschitz | ⚠️ Fixed margin | Safe but conservative (deadlocks) |
| **CR-PH-CBF (Ours)** | ✅ Probabilistic (99%) | ✅ Adaptive conformal | Best balance: safe + performant |

---

## Configuration

Key parameters can be modified in `src/utils/config.py`:

```python
class ExperimentConfig:
    SEED = 42                    # Random seed
    NUM_EPISODES = 20            # Episodes per method
    MAX_STEPS = 250              # Steps per episode
    UNCERTAINTY_LEVEL = "high"   # "low", "medium", or "high"
    
    CBF_GAINS = {'k1': 2.0, 'k2': 2.0}  # Class-K gains
    CONFORMAL_DELTA = 0.01       # 99% coverage
    LEARNING_RATE = 1e-3         # PH-ResNet training
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2026crphcbf,
  title={Conformal Robust Port-Hamiltonian Control Barrier Filters for Safe Robot Control Under Uncertainty},
  author={Your Name and Collaborators},
  booktitle={Conference Name},
  year={2026}
}
```

---

## Technical Details

### Port-Hamiltonian Structure

The PH-ResNet enforces energy conservation through:
- **Conservative forces**: \( F_{\text{cons}} = -\nabla V_\theta(q) \)
- **Dissipative forces**: \( F_{\text{diss}} = -D_\theta(q, \dot{q}) \dot{q} \), where \( D = LL^T \geq 0 \)

This guarantees \( \dot{H} \leq 0 \) (passivity), ensuring stability.

### High-Order CBF Formulation

For relative-degree 2 barriers:
```
b₁ = dh/dt + k₁h ≥ 0
db₁/dt ≥ -k₂b₁ - ε_t||A_cbf|| + A_cbf·f_pred
```

where `ε_t` is the adaptive conformal margin.

### Adaptive Conformal Calibration

Online quantile tracking:
```
ε_t ← max(ε_min, ε_t + β·(𝟙{e_t > ε_t} - δ))
```

This provides probabilistic coverage: \( P(|e| ≤ ε_t) ≥ 1 - δ \).

---

## Reproducing Paper Results

The code is configured to exactly reproduce the results in **Table I** of the paper:

```bash
# Ensure random seed is fixed (SEED=42 in config.py)
python experiments/run_benchmark.py
```

All experiments use:
- 20 episodes per method
- High uncertainty (166% payload increase, 3× friction, 4N wind)
- 250 timesteps per episode (2.5 seconds at 100 Hz)
- Fixed initial conditions

---

## Dependencies

- `numpy>=1.21.0` - Numerical computation
- `torch>=1.10.0` - Neural network training
- `scipy>=1.7.0` - QP solver (SLSQP)

See `requirements.txt` for complete list.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@institution.edu
- Lab Website: https://yourlab.edu

---

## Acknowledgments

This work builds upon:
- **Control Barrier Functions**: Ames et al. (2019)
- **Robust CBF**: Taylor et al. (2020)
- **Port-Hamiltonian Systems**: van der Schaft & Jeltsema (2014)
- **Conformal Prediction**: Vovk et al. (2005)
