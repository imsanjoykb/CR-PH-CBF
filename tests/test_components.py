"""
Unit tests for core components
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crphcbf.environments import TwoLinkRobotEnv
from crphcbf.models import PHResNet
from crphcbf.controllers import SafeCBFFilter
from crphcbf.utils import ExperimentConfig


def test_environment():
    """Test robot environment dynamics."""
    print("Testing TwoLinkRobotEnv...")
    env = TwoLinkRobotEnv(uncertainty_level="high")
    
    # Test reset
    state = env.reset()
    assert state.shape == (4,), "State shape mismatch"
    
    # Test step
    u = np.array([1.0, 1.0])
    next_state, reward, h, ddq = env.step(u)
    assert next_state.shape == (4,), "Next state shape mismatch"
    assert isinstance(reward, (float, np.floating)), "Reward should be scalar"
    assert isinstance(h, (float, np.floating)), "Barrier should be scalar"
    assert ddq.shape == (2,), "Acceleration shape mismatch"
    
    # Test kinematics
    q = np.array([0.0, 0.0])
    p_ee = env.forward_kinematics(q)
    assert p_ee.shape == (2,), "EE position shape mismatch"
    assert np.allclose(p_ee, [2.0, 0.0], atol=1e-6), "FK at zero config incorrect"
    
    J = env.jacobian(q)
    assert J.shape == (2, 2), "Jacobian shape mismatch"
    
    print("✓ Environment tests passed\n")


def test_ph_model():
    """Test Port-Hamiltonian neural network."""
    print("Testing PHResNet...")
    model = PHResNet(hidden_dim=64)
    
    # Test forward pass
    q = torch.randn(5, 2)
    dq = torch.randn(5, 2)
    F = model(q, dq)
    
    assert F.shape == (5, 2), "Output shape mismatch"
    assert not torch.isnan(F).any(), "NaN in output"
    
    # Test gradient computation
    loss = F.sum()
    loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed"
        assert not torch.isnan(param.grad).any(), "NaN in gradients"
    
    print("✓ PH-ResNet tests passed\n")


def test_cbf_controller():
    """Test CBF safety filter."""
    print("Testing SafeCBFFilter...")
    env = TwoLinkRobotEnv(uncertainty_level="medium")
    model = PHResNet()
    cbf_filter = SafeCBFFilter(env, model, k1=2.0, k2=2.0)
    
    # Test QP solving
    state = env.reset()
    u_nom = np.array([5.0, 5.0])
    
    for mode in ["nominal", "conservative_robust", "cr_ph_cbf"]:
        u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode=mode)
        assert u_safe.shape == (2,), f"Safe control shape mismatch for {mode}"
        assert isinstance(feasible, bool), f"Feasibility should be bool for {mode}"
        assert np.all(np.abs(u_safe) <= 25.1), f"Torque limits violated for {mode}"
    
    # Test conformal update
    cbf_filter.update_conformal_margin(0.5)
    assert len(cbf_filter.cal_residuals) == 1, "Residual not stored"
    
    print("✓ CBF controller tests passed\n")


def test_integration():
    """Test full integration with short episode."""
    print("Testing full integration...")
    config = ExperimentConfig()
    config.set_seed()
    
    env = TwoLinkRobotEnv(uncertainty_level="low")
    model = PHResNet()
    cbf_filter = SafeCBFFilter(env, model)
    
    state = env.reset()
    
    # Run 10 steps
    for _ in range(10):
        q = state[:2]
        dq = state[2:]
        u_nom = 10.0 * (env.goal_q - q) - 2.0 * dq
        u_safe, _ = cbf_filter.solve_qp(state, u_nom, mode="cr_ph_cbf")
        state, _, _, _ = env.step(u_safe)
    
    print("✓ Integration test passed\n")


if __name__ == "__main__":
    print("="*60)
    print("Running CR-PH-CBF Test Suite")
    print("="*60 + "\n")
    
    test_environment()
    test_ph_model()
    test_cbf_controller()
    test_integration()
    
    print("="*60)
    print("All tests passed!")
    print("="*60)
