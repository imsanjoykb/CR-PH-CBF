#!/usr/bin/env python3
"""
Quick Start Script - Run this to verify everything works
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    missing = []
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy - MISSING")
    
    try:
        import torch
        print("✓ torch")
    except ImportError:
        missing.append("torch")
        print("✗ torch - MISSING")
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError:
        missing.append("scipy")
        print("✗ scipy - MISSING")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed!\n")
    return True


def quick_test():
    """Run a quick functionality test."""
    print("Running quick functionality test...\n")
    
    from crphcbf.environments import TwoLinkRobotEnv
    from crphcbf.models import PHResNet
    from crphcbf.controllers import SafeCBFFilter
    
    # Test environment
    print("1. Testing environment...")
    env = TwoLinkRobotEnv(uncertainty_level="low")
    state = env.reset()
    state, _, _, _ = env.step([1.0, 1.0])
    print("   ✓ Environment working\n")
    
    # Test model
    print("2. Testing Port-Hamiltonian model...")
    import torch
    model = PHResNet(hidden_dim=32)
    q = torch.randn(1, 2)
    dq = torch.randn(1, 2)
    output = model(q, dq)
    print("   ✓ Model working\n")
    
    # Test controller
    print("3. Testing CBF controller...")
    import numpy as np
    controller = SafeCBFFilter(env, model)
    state = env.reset()
    u_nom = np.array([5.0, 5.0])
    u_safe, feasible = controller.solve_qp(state, u_nom, mode="cr_ph_cbf")
    print("   ✓ Controller working\n")
    
    print("="*60)
    print("Quick test PASSED! 🎉")
    print("="*60)
    print("\nYou're ready to run experiments!")
    print("Try: python3 experiments/run_benchmark.py")


def main():
    print("="*60)
    print("CR-PH-CBF Quick Start")
    print("="*60 + "\n")
    
    if not check_dependencies():
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Try: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")
        sys.exit(1)


if __name__ == "__main__":
    main()
