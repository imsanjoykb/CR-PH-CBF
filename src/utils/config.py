"""
Experiment Configuration
========================
Central configuration management for reproducible experiments.
"""

import numpy as np
import torch


class ExperimentConfig:
    """
    Configuration parameters for CR-PH-CBF experiments.
    
    Attributes
    ----------
    SEED : int
        Random seed for reproducibility
    DT : float
        Control timestep (100 Hz)
    NUM_EPISODES : int
        Number of evaluation episodes per method
    MAX_STEPS : int
        Maximum steps per episode
    UNCERTAINTY_LEVEL : str
        Dynamics uncertainty: "low", "medium", or "high"
    CBF_GAINS : dict
        Class-K function gains for HOCBF
    TORQUE_LIMITS : tuple
        Joint torque bounds (N·m)
    """
    
    # Reproducibility
    SEED = 42
    
    # Simulation parameters
    DT = 0.01  # 100 Hz control
    NUM_EPISODES = 20
    MAX_STEPS = 250
    UNCERTAINTY_LEVEL = "high"
    
    # Control parameters
    CBF_GAINS = {
        'k1': 2.0,  # Position barrier gain
        'k2': 2.0   # Velocity barrier gain
    }
    
    TORQUE_LIMITS = (-25.0, 25.0)
    
    # PD Controller gains
    PD_GAINS = {
        'kp': 15.0,
        'kd': 3.0
    }
    
    # Conformal prediction
    CONFORMAL_DELTA = 0.01  # 99% coverage
    CONFORMAL_BETA = 0.05   # Adaptation rate
    CONFORMAL_WINDOW = 200  # Calibration window
    
    # Training parameters
    TRAINING_ITERATIONS = 200
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 64
    
    # Evaluation modes
    MODES = [
        "unconstrained",
        "nominal_cbf",
        "conservative_robust",
        "cr_ph_cbf"
    ]
    
    @classmethod
    def set_seed(cls):
        """Set random seeds for reproducibility."""
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
