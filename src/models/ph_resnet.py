"""
Port-Hamiltonian Residual Neural Network (PH-ResNet)
====================================================
Physics-informed neural architecture that enforces passivity constraints
for learning unmodeled dynamics in robot systems.

Reference:
    Incorporates energy-based structure from Port-Hamiltonian system theory
    to guarantee that learned residuals preserve system stability.
"""

import torch
import torch.nn as nn


class PHResNet(nn.Module):
    """
    Port-Hamiltonian Residual Neural Network.
    
    Enforces passivity constraint: dH/dt <= 0 by factoring unmodeled forces into:
    1. Conservative forces: F_cons = -∇V_θ(q) (gradient of learned potential)
    2. Dissipative forces: F_diss = -D_θ(q, dq) * dq (Cholesky-factored damping)
    
    This structure guarantees that the learned model does not inject energy,
    preserving closed-loop stability properties.
    
    Architecture
    ------------
    - Potential Network V_θ(q): Maps joint positions to scalar potential energy
    - Dissipation Network L_D(q, dq): Outputs lower-triangular Cholesky factor
      for positive semi-definite damping matrix D = L * L^T >= 0
    
    Parameters
    ----------
    hidden_dim : int, optional
        Hidden layer dimension (default: 64)
    """
    
    def __init__(self, hidden_dim=64):
        super(PHResNet, self).__init__()
        
        # Scalar potential network V_theta(q)
        self.potential_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Dissipation Cholesky factor network L_D(q, dq)
        # Outputs [L11, L21, L22] for 2x2 lower-triangular matrix
        self.dissipation_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, q, dq):
        """
        Compute Port-Hamiltonian residual forces.
        
        Parameters
        ----------
        q : torch.Tensor, shape (batch_size, 2)
            Joint positions [q1, q2]
        dq : torch.Tensor, shape (batch_size, 2)
            Joint velocities [dq1, dq2]
            
        Returns
        -------
        F_total : torch.Tensor, shape (batch_size, 2)
            Total residual force = F_cons + F_diss
        """
        # 1. Conservative force: F_cons = -∇_q V_θ
        q_grad = q.clone().detach().requires_grad_(True)
        V = self.potential_net(q_grad)
        grad_V = torch.autograd.grad(V.sum(), q_grad, create_graph=True)[0]
        F_cons = -grad_V
        
        # 2. Non-negative Rayleigh Dissipation: D_θ = L * L^T >= 0
        state_4d = torch.cat([q, dq], dim=-1)
        L_params = self.dissipation_net(state_4d)
        
        # Enforce positive diagonal via exponential
        L11 = torch.exp(L_params[:, 0:1])  # strictly positive
        L21 = L_params[:, 1:2]
        L22 = torch.exp(L_params[:, 2:3])
        
        # Compute D * dq where D = [[L11^2, L11*L21], [L11*L21, L21^2 + L22^2]]
        D11 = L11**2
        D12 = L11 * L21
        D22 = L21**2 + L22**2
        
        fric_1 = D11 * dq[:, 0:1] + D12 * dq[:, 1:2]
        fric_2 = D12 * dq[:, 0:1] + D22 * dq[:, 1:2]
        F_diss = -torch.cat([fric_1, fric_2], dim=-1)
        
        return F_cons + F_diss
