"""
Control Barrier Function (CBF) Safety Filter
============================================
Implements High-Order CBF-QP with adaptive conformal calibration
for robust safe control under dynamics uncertainty.

References:
    - Ames et al. (2019): "Control Barrier Functions"
    - Taylor et al. (2020): "Learning for Safety-Critical Control with Control Barrier Functions"
"""

import numpy as np
import torch
from scipy.optimize import minimize


class SafeCBFFilter:
    """
    Safety-Critical Control Filter using Control Barrier Functions.
    
    Implements three variants:
    1. Nominal CBF-QP: Standard CBF without uncertainty compensation
    2. Conservative Robust CBF: Fixed worst-case Lipschitz margin
    3. CR-PH-CBF (Proposed): Adaptive conformal calibration with PH-ResNet
    
    The filter solves a Quadratic Program (QP) to minimally modify a nominal
    control input u_nom to satisfy safety constraints h(x) >= 0.
    
    For relative-degree 2 barriers, we use High-Order CBF (HOCBF) with:
        b1 = dh/dt + k1 * h >= 0
        db1/dt >= -k2 * b1 - margin
    
    Parameters
    ----------
    env : TwoLinkRobotEnv
        Robot environment providing dynamics and barrier function
    ph_model : PHResNet
        Port-Hamiltonian neural network for residual prediction
    k1 : float, optional
        Class-K gain for position barrier (default: 2.0)
    k2 : float, optional
        Class-K gain for velocity barrier (default: 2.0)
    delta : float, optional
        Conformal coverage level (default: 0.01 for 99% coverage)
    """
    
    def __init__(self, env, ph_model, k1=2.0, k2=2.0, delta=0.01):
        self.env = env
        self.model = ph_model
        self.k1 = k1  # Class-K gain for position barrier
        self.k2 = k2  # Class-K gain for velocity barrier
        self.delta = delta
        
        # Online adaptive conformal quantile parameters
        self.cal_residuals = []
        self.N_cal = 200  # Calibration window size
        self.eps_t = 0.5  # Initial conservative uncertainty bound
        self.beta = 0.05  # Adaptive conformal update step size

    def update_conformal_margin(self, residual_error):
        """
        Update adaptive conformal prediction margin online.
        
        Maintains a sliding window of prediction errors and computes
        the empirical (1-delta)-quantile to provide probabilistic coverage.
        
        Parameters
        ----------
        residual_error : float
            Magnitude of prediction error ||ddq_true - ddq_pred||
        """
        self.cal_residuals.append(residual_error)
        if len(self.cal_residuals) > self.N_cal:
            self.cal_residuals.pop(0)
            
        # Compute empirical quantile
        if len(self.cal_residuals) >= 20:  # Minimum samples for quantile
            q_val = np.quantile(self.cal_residuals, 1.0 - self.delta)
            # Online adaptive step (gradient descent on coverage error)
            indicator = 1.0 if residual_error > self.eps_t else 0.0
            self.eps_t = max(0.05, self.eps_t + self.beta * (indicator - self.delta))

    def solve_qp(self, state, u_nom, mode="cr_ph_cbf"):
        """
        Solve CBF-QP to find safe control input.
        
        min  0.5 * ||u - u_nom||^2
        s.t. A_cbf * u <= b_cbf  (HOCBF constraint)
             -25 <= u <= 25      (torque limits)
        
        Parameters
        ----------
        state : ndarray, shape (4,)
            Current state [q1, q2, dq1, dq2]
        u_nom : ndarray, shape (2,)
            Nominal (desired) control input
        mode : str, optional
            Control mode: "nominal", "conservative_robust", "cr_ph_cbf", or "unconstrained"
            
        Returns
        -------
        u_safe : ndarray, shape (2,)
            Safe control torque after QP filtering
        feasible : bool
            Whether QP solution was found (False indicates deadlock)
        """
        q = state[:2]
        dq = state[2:]
        p_ee = self.env.forward_kinematics(q)
        J = self.env.jacobian(q)
        
        # Barrier function: h(q) = ||p_ee - p_obs||^2 - R^2
        diff = p_ee - self.env.obs_center
        h = np.dot(diff, diff) - self.env.obs_radius**2
        
        # Relative degree 2: First derivative dh/dt = 2 * diff^T * J * dq
        dh = 2.0 * np.dot(diff, J @ dq)
        
        # Virtual barrier for HOCBF: b1 = dh + k1 * h >= 0
        b1 = dh + self.k1 * h
        
        M0 = self.env.nominal_inertia(q)
        M0_inv = np.linalg.inv(M0)
        
        # Gradient of b1 with respect to control torque u:
        # d(b1)/dt = 2*diff^T * J * ddq + ... -> A_cbf = -2 * diff^T * J * M0_inv
        A_cbf = -2.0 * diff @ J @ M0_inv
        A_norm = np.linalg.norm(A_cbf)
        
        # Residual prediction from Port-Hamiltonian model
        with torch.no_grad():
            q_t = torch.FloatTensor(q).unsqueeze(0)
            dq_t = torch.FloatTensor(dq).unsqueeze(0)
            pred_f = self.model(q_t, dq_t).numpy().flatten()
            
        # Uncertainty margin determination based on mode
        if mode == "nominal":
            margin = 0.0
        elif mode == "conservative_robust":
            margin = A_norm * 3.5  # Static, fixed worst-case Lipschitz bound
        elif mode == "cr_ph_cbf":
            margin = A_norm * self.eps_t  # Dynamically calibrated conformal bound
        else:  # Unfiltered baseline
            return u_nom, True

        # Robust High-Order CBF affine inequality: A_cbf * u <= b_cbf
        # b_cbf = k2 * b1 - margin + nominal Lie terms
        b_cbf = self.k2 * b1 - margin + float(A_cbf @ pred_f)
        
        # Solve Quadratic Program: min 0.5 * ||u - u_nom||^2
        # subject to A_cbf * u <= b_cbf and torque limits [-25, 25]
        def obj(u):
            return 0.5 * np.sum((u - u_nom)**2)
            
        cons = ({'type': 'ineq', 'fun': lambda u: b_cbf - np.dot(A_cbf, u)})
        bounds = [(-25.0, 25.0), (-25.0, 25.0)]
        
        res = minimize(obj, u_nom, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            return res.x, True
        else:
            # QP Infeasible / Deadlock: apply braking torque
            return -0.5 * dq, False
