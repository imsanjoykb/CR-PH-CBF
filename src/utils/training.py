"""
Port-Hamiltonian Model Training Utilities
=========================================
Functions for pre-training the PH-ResNet on exploratory data.
"""

import numpy as np
import torch
import torch.optim as optim


def train_ph_model(ph_model, env, num_iterations=200, lr=1e-3, verbose=True):
    """
    Pre-train Port-Hamiltonian residual model on exploratory data.
    
    Collects random state-action pairs and learns to predict acceleration
    residuals between nominal and true dynamics.
    
    Parameters
    ----------
    ph_model : PHResNet
        Port-Hamiltonian neural network to train
    env : TwoLinkRobotEnv
        Robot environment for generating training data
    num_iterations : int, optional
        Number of training iterations (default: 200)
    lr : float, optional
        Learning rate (default: 1e-3)
    verbose : bool, optional
        Print training progress (default: True)
        
    Returns
    -------
    losses : list of float
        Training loss history
    """
    optimizer = optim.Adam(ph_model.parameters(), lr=lr)
    losses = []
    
    if verbose:
        print(f"Pre-training Port-Hamiltonian model ({num_iterations} iterations)...")
    
    for i in range(num_iterations):
        # Sample random state-action pair
        q = np.random.uniform(-np.pi, np.pi, 2)
        dq = np.random.uniform(-2.0, 2.0, 2)
        u = np.random.uniform(-10.0, 10.0, 2)
        
        # Compute nominal dynamics prediction
        M0 = env.nominal_inertia(q)
        ddq_nom = np.linalg.inv(M0) @ (u - env.d_nom * dq)
        
        # Add measurement noise to simulate real sensor data
        ddq_meas = ddq_nom + np.random.normal(0, 0.1, 2)
        
        # Convert to tensors
        q_t = torch.FloatTensor(q).unsqueeze(0)
        dq_t = torch.FloatTensor(dq).unsqueeze(0)
        ddq_target = torch.FloatTensor(ddq_meas)
        
        # Forward pass and loss computation
        ddq_pred = ph_model(q_t, dq_t).squeeze(0)
        loss = torch.mean((ddq_pred - ddq_target)**2)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (i + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"  Iteration {i+1}/{num_iterations}, Loss: {avg_loss:.6f}")
    
    if verbose:
        print(f"Training complete. Final loss: {np.mean(losses[-10]):.6f}\n")
    
    return losses
