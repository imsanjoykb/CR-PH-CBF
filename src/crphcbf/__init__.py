"""
CR-PH-CBF: Conformal Robust Port-Hamiltonian Control Barrier Filters
=====================================================================

A research package for safe robot control under dynamics uncertainty.

Modules:
    environments: Robot simulation environments
    models: Neural network architectures (PH-ResNet)
    controllers: CBF-QP safety filters
    utils: Training and configuration utilities

Example:
    >>> from crphcbf.environments import TwoLinkRobotEnv
    >>> from crphcbf.models import PHResNet
    >>> from crphcbf.controllers import SafeCBFFilter
    >>> 
    >>> env = TwoLinkRobotEnv(uncertainty_level="high")
    >>> model = PHResNet()
    >>> controller = SafeCBFFilter(env, model)

Authors: Your Research Team
License: MIT
Paper: https://arxiv.org/abs/2406.xxxxx
"""

__version__ = "1.0.0"
__author__ = "Your Research Team"
__license__ = "MIT"
__paper__ = "https://arxiv.org/abs/2406.xxxxx"

from .environments import TwoLinkRobotEnv
from .models import PHResNet
from .controllers import SafeCBFFilter
from .utils import ExperimentConfig, train_ph_model

__all__ = [
    'TwoLinkRobotEnv',
    'PHResNet',
    'SafeCBFFilter',
    'ExperimentConfig',
    'train_ph_model',
]
