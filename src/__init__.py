"""
CR-PH-CBF: Conformal Robust Port-Hamiltonian Control Barrier Filters
=====================================================================

Main package module providing convenient top-level imports.

Example:
    >>> from crphcbf import TwoLinkRobotEnv, PHResNet, SafeCBFFilter
    >>> env = TwoLinkRobotEnv(uncertainty_level="high")
    >>> model = PHResNet()
    >>> controller = SafeCBFFilter(env, model)
"""

from .crphcbf import *

__version__ = "1.0.0"
__author__ = "Your Research Team"
__license__ = "MIT"
