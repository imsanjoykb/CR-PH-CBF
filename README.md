# Conformal Robust Port-Hamiltonian Control Barrier Filters

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2406.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2406.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

**[Paper](https://arxiv.org/abs/2406.xxxxx)** | **[Project Page](https://yourlab.edu/cr-ph-cbf)** | **[Video](https://youtube.com/xxxxx)** | **[Poster](paper/poster.pdf)**

</div>

---

## 📖 About

Official PyTorch implementation of **"Conformal Robust Port-Hamiltonian Control Barrier Filters for Safe Robot Control Under Dynamics Uncertainty"**.

**TL;DR:** We combine *Port-Hamiltonian neural networks* with *adaptive conformal prediction* to achieve **provably safe** robot control under severe dynamics uncertainty, outperforming nominal and conservative baselines by **60% in task success** while maintaining **99% safety coverage**.

<p align="center">
  <img src="assets/method_overview.png" alt="Method Overview" width="800"/>
</p>

### 🎯 Key Contributions

1. **Port-Hamiltonian ResNet (PH-ResNet)** - Physics-informed neural architecture that guarantees energy passivity (∇H ≤ 0) for learning unmodeled dynamics
2. **Adaptive Conformal Calibration** - Online uncertainty quantification with probabilistic coverage guarantees
3. **High-Order CBF-QP Framework** - Robust safety filter combining learned dynamics with formal barrier certificates
4. **Rigorous Empirical Validation** - Benchmark on 2-DOF manipulator with 166% payload uncertainty, 3× friction, and external disturbances

---

## 📰 News

- **[2026-10]** 🎉 Paper accepted at **Conference Name 2026**!
- **[2026-09]** Code and models released
- **[2026-06]** Paper submitted to arXiv

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CR-PH-CBF.git
cd CR-PH-CBF

# Create conda environment
conda create -n crphcbf python=3.8
conda activate crphcbf

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Reproduce main paper results (Table 1)
python scripts/reproduce_paper_results.py

# Run ablation studies
python scripts/run_ablations.py

# Train custom models
python scripts/train.py --config configs/high_uncertainty.yaml
```

### Expected Output

```
═══════════════════════════════════════════════════════════════════════════════
                          Table 1: Main Results (High Uncertainty)
═══════════════════════════════════════════════════════════════════════════════
Method                  Success Rate ↑   Violations ↓   Max Penetration ↓   Deadlock Rate ↓
─────────────────────────────────────────────────────────────────────────────────────────────
Unconstrained                    0.0%          24.1              0.156              0.0%
Nominal CBF                     35.0%           8.3              0.042              2.1%
Conservative Robust             75.0%           0.5              0.003             18.4%
CR-PH-CBF (Ours)               95.0%           0.1              0.001              3.2%
═══════════════════════════════════════════════════════════════════════════════
```

---

## 📂 Repository Structure

```
CR-PH-CBF/
│
├── 📄 Paper & Assets
│   ├── paper/
│   │   ├── figures/              # All paper figures (PDF/PNG)
│   │   ├── tables/               # LaTeX table source
│   │   ├── data/                 # Numerical results (CSV/JSON)
│   │   └── supplementary.pdf     # Supplementary material
│   └── assets/                   # README images & diagrams
│
├── 🔬 Source Code
│   ├── src/
│   │   ├── crphcbf/
│   │   │   ├── environments/     # Robot simulation environments
│   │   │   ├── models/           # Neural network architectures
│   │   │   ├── controllers/      # CBF-QP safety filters
│   │   │   └── utils/            # Helper functions
│   │   └── __init__.py
│
├── 🧪 Experiments
│   ├── scripts/
│   │   ├── reproduce_paper_results.py
│   │   ├── run_ablations.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── configs/                  # Experiment configurations (YAML)
│
├── 📊 Analysis
│   ├── notebooks/                # Jupyter notebooks
│   │   ├── 01_data_analysis.ipynb
│   │   ├── 02_visualization.ipynb
│   │   └── 03_ablation_study.ipynb
│   └── results/                  # Saved experiment outputs
│
├── ✅ Testing
│   ├── tests/
│   │   ├── test_environments.py
│   │   ├── test_models.py
│   │   └── test_controllers.py
│   └── pytest.ini
│
├── 🐳 Deployment
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── environment.yml           # Conda environment file
│
└── 📚 Documentation
    ├── docs/
    │   ├── installation.md
    │   ├── api/                  # API documentation
    │   ├── tutorials/            # Step-by-step guides
    │   └── faq.md
    ├── CONTRIBUTING.md
    ├── CODE_OF_CONDUCT.md
    ├── LICENSE
    └── CITATION.cff              # Citation metadata
```

---

## 🧠 Method Overview

### Port-Hamiltonian Neural Network

Our PH-ResNet enforces energy-preserving structure:

```math
F_{res}(q, \dot{q}) = \underbrace{-\nabla V_\theta(q)}_{\text{Conservative}} + \underbrace{(-D_\theta(q, \dot{q}) \dot{q})}_{\text{Dissipative}}
```

where D_θ = LL^T ≥ 0 guarantees passivity: ∂H/∂t ≤ 0.

### Adaptive Conformal Calibration

Online uncertainty quantification:

```math
\epsilon_t \leftarrow \max(\epsilon_{min}, \epsilon_t + \beta \cdot (\mathbb{1}\{e_t > \epsilon_t\} - \delta))
```

Provides (1-δ)-coverage: P(|e| ≤ ε_t) ≥ 1 - δ.

### High-Order CBF Constraint

For relative-degree 2 barriers:

```math
\dot{b}_1 \geq -k_2 b_1 - \epsilon_t \|A_{cbf}\| + A_{cbf} \cdot f_{pred}
```

where b₁ = ∂h/∂t + k₁h ensures forward invariance of safe set h(x) ≥ 0.

---

## 📊 Results

### Main Comparison (Table 1)

| Method | Success Rate | Safety Violations | Computational Cost |
|--------|--------------|-------------------|-------------------|
| Unconstrained | 0.0% | High | Low |
| Nominal CBF | 35.0% | Medium | Medium |
| Conservative Robust | 75.0% | Low | High (18% deadlock) |
| **CR-PH-CBF (Ours)** | **95.0%** | **Minimal** | **Balanced** |

### Ablation Studies

<p align="center">
  <img src="paper/figures/ablation_uncertainty.png" alt="Ablation" width="600"/>
</p>

See `paper/tables/` for complete numerical results and LaTeX source.

---

## 🔧 Advanced Usage

### Custom Environments

```python
from crphcbf.environments import TwoLinkRobotEnv

# Create custom environment
env = TwoLinkRobotEnv(
    uncertainty_level="custom",
    payload_mass=2.5,
    friction_multiplier=3.0,
    external_force=4.0
)
```

### Training Your Own Models

```python
from crphcbf.models import PHResNet
from crphcbf.utils import train_ph_model

# Initialize model
model = PHResNet(hidden_dim=128, activation='tanh')

# Train on custom data
train_ph_model(
    model, env,
    num_iterations=500,
    learning_rate=1e-3,
    save_path='checkpoints/my_model.pt'
)
```

### Custom Safety Specifications

```python
from crphcbf.controllers import SafeCBFFilter

# Configure controller
controller = SafeCBFFilter(
    env, model,
    cbf_gains={'k1': 3.0, 'k2': 3.0},
    conformal_delta=0.01,  # 99% coverage
    conformal_beta=0.05
)
```

---

## 📈 Reproducing Paper Results

All experiments are fully reproducible with fixed seeds:

```bash
# Main results (Table 1)
python scripts/reproduce_paper_results.py --table 1

# Ablation study (Figure 3)
python scripts/run_ablations.py --figure 3

# All paper figures and tables
bash scripts/reproduce_all.sh
```

Results are saved to `results/paper/` with timestamps and git commit hashes for full traceability.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_models.py -v

# Check code coverage
pytest --cov=src --cov-report=html
```

---

## 🐳 Docker

```bash
# Build image
docker build -t crphcbf:latest -f docker/Dockerfile .

# Run experiments in container
docker-compose up

# Interactive session
docker run -it --rm -v $(pwd):/workspace crphcbf:latest bash
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yourname2026crphcbf,
  title={Conformal Robust Port-Hamiltonian Control Barrier Filters for Safe Robot Control Under Dynamics Uncertainty},
  author={Your Name and Co-authors},
  booktitle={Proceedings of Conference Name},
  year={2026},
  url={https://arxiv.org/abs/2406.xxxxx}
}
```

**BibTeX file**: [`CITATION.bib`](CITATION.bib)  
**Citation metadata**: [`CITATION.cff`](CITATION.cff)

---

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

- Report bugs via [GitHub Issues](https://github.com/yourusername/CR-PH-CBF/issues)
- Submit improvements via [Pull Requests](https://github.com/yourusername/CR-PH-CBF/pulls)
- Join discussions in [GitHub Discussions](https://github.com/yourusername/CR-PH-CBF/discussions)

---

## 📜 License

This project is licensed under the MIT License - see [`LICENSE`](LICENSE) for details.

---

## 👥 Authors

- **Your Name** - [Lab/University](https://yourlab.edu) - [Email](mailto:you@domain.edu)
- **Co-author Name** - [Affiliation](https://affiliation.edu)

**Lab**: [Robotics and AI Lab](https://yourlab.edu)  
**Institution**: University Name

---

## 🙏 Acknowledgments

This work was supported by:
- Grant Agency (Grant Number: XXX-XXXX)
- Computing resources from SuperComputer Center

We thank the following open-source projects:
- [PyTorch](https://pytorch.org/) for deep learning framework
- [SciPy](https://scipy.org/) for optimization
- [matplotlib](https://matplotlib.org/) for visualization

---

## 📞 Contact

For questions or collaborations, please contact:
- Email: your.email@university.edu
- Lab Website: https://yourlab.edu
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

<div align="center">

**[⬆ Back to Top](#conformal-robust-port-hamiltonian-control-barrier-filters)**

Made with ❤️ by the Robotics Lab @ University Name

</div>
