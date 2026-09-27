import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import time

# Set deterministic random seeds for full reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. 2-DOF MANIPULATOR DYNAMICS & UNCERTAIN ENVIRONMENT
class TwoLinkRobotEnv:
    def __init__(self, uncertainty_level="high"):
        self.dt = 0.01  # 100 Hz control loop
        self.l1 = 1.0   # Link 1 length (m)
        self.l2 = 1.0   # Link 2 length (m)
        
        # Nominal parameters (assumed by nominal controller)
        self.m1_nom = 1.5   # Link 1 mass (kg)
        self.m2_nom = 1.5   # Link 2 mass (kg)
        self.d_nom = np.array([0.2, 0.2])  # Nominal joint damping
        
        # Real perturbed parameters
        self.uncertainty_level = uncertainty_level
        if uncertainty_level == "low":
            self.m_payload = 0.1   # +6% payload
            self.friction_mult = 1.1
            self.wind_force = 0.0
        elif uncertainty_level == "medium":
            self.m_payload = 1.5   # +100% payload
            self.friction_mult = 2.0
            self.wind_force = 1.5
        elif uncertainty_level == "high":
            self.m_payload = 2.5   # +166% payload mass shift
            self.friction_mult = 3.0
            self.wind_force = 4.0  # Constant external drag force (N)
            
        self.m1_true = self.m1_nom
        self.m2_true = self.m2_nom + self.m_payload
        self.g = 9.81
        
        # Obstacle Keep-Out Zone (Cartesian End-Effector Space)
        self.obs_center = np.array([1.1, 0.4])
        self.obs_radius = 0.35  # Safety boundary R
        
        # Goal configuration
        self.goal_q = np.array([np.pi/3, -np.pi/4])
        self.reset()

    def reset(self):
        # Start state [q1, q2, dq1, dq2]
        self.state = np.array([0.05, 0.1, 0.0, 0.0])
        return self.state.copy()

    def forward_kinematics(self, q):
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def jacobian(self, q):
        J = np.array([
            [-self.l1 * np.sin(q[0]) - self.l2 * np.sin(q[0] + q[1]), -self.l2 * np.sin(q[0] + q[1])],
            [ self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]),  self.l2 * np.cos(q[0] + q[1])]
        ])
        return J

    def nominal_inertia(self, q):
        a1 = 3.0 + 1.5
        a2 = 1.5 * np.cos(q[1])
        a3 = 1.5
        M0 = np.array([[a1 + 2*a2, a3 + a2],
                       [a3 + a2,    a3]])
        return M0

    def true_inertia(self, q):
        m2 = self.m2_true
        a1 = (self.m1_true + m2) * self.l1**2 + m2 * self.l2**2
        a2 = m2 * self.l1 * self.l2 * np.cos(q[1])
        a3 = m2 * self.l2**2
        M = np.array([[a1 + 2*a2, a3 + a2],
                      [a3 + a2,    a3]])
        return M

    def step(self, u):
        q = self.state[:2]
        dq = self.state[2:]
        
        # Compute true non-linear continuous physics
        M = self.true_inertia(q)
        M_inv = np.linalg.inv(M)
        
        # Gravity & non-conservative forces
        g_true = np.array([
            (self.m1_true*0.5 + self.m2_true)*self.g*np.cos(q[0]) + self.m2_true*self.g*np.cos(q[0]+q[1]),
            self.m2_true*self.g*np.cos(q[0]+q[1])
        ])
        
        # Joint friction stiction + viscous
        fric = self.friction_mult * (self.d_nom * dq + 0.5 * np.sign(dq))
        
        # Wind perturbation on end-effector transformed via J^T
        J = self.jacobian(q)
        tau_wind = J.T @ np.array([self.wind_force, -self.wind_force * 0.5])
        
        # Acceleration ddq = M^-1 * (u - g - fric + tau_wind)
        ddq = M_inv @ (u - g_true - fric + tau_wind)
        
        # Semi-implicit Euler integration
        dq_new = dq + ddq * self.dt
        q_new = q + dq_new * self.dt
        self.state = np.concatenate([q_new, dq_new])
        
        # Reward: progress toward goal minus control effort
        dist_to_goal = np.linalg.norm(q_new - self.goal_q)
        reward = -dist_to_goal - 0.01 * np.sum(u**2)
        
        # Safety Barrier Metric h(q) = ||p_ee - p_obs||^2 - R^2
        p_ee = self.forward_kinematics(q_new)
        dist_obs = np.linalg.norm(p_ee - self.obs_center)
        h = dist_obs**2 - self.obs_radius**2
        
        return self.state.copy(), reward, h, ddq

# 2. PORT-HAMILTONIAN RESIDUAL NEURAL NETWORK (PH-ResNet)
class PHResNet(nn.Module):
    def __init__(self):
        super(PHResNet, self).__init__()
        # Scalar potential network V_theta(q)
        self.potential_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Dissipation Cholesky factor network L_D(q, dq)
        self.dissipation_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Lower-triangular 2x2 elements: [L11, L21, L22]
        )

    def forward(self, q, dq):
        # 1. Conservative force F_cons = -nabla_q V_theta
        q_grad = q.clone().detach().requires_grad_(True)
        V = self.potential_net(q_grad)
        grad_V = torch.autograd.grad(V.sum(), q_grad, create_graph=True)[0]
        F_cons = -grad_V
        
        # 2. Non-negative Rayleigh Dissipation D_theta = L * L^T >= 0
        state_4d = torch.cat([q, dq], dim=-1)
        L_params = self.dissipation_net(state_4d)
        L11 = torch.exp(L_params[:, 0:1])  # strictly positive diagonal
        L21 = L_params[:, 1:2]
        L22 = torch.exp(L_params[:, 2:3])
        
        # Form product D * dq
        D11 = L11**2
        D12 = L11 * L21
        D22 = L21**2 + L22**2
        
        fric_1 = D11 * dq[:, 0:1] + D12 * dq[:, 1:2]
        fric_2 = D12 * dq[:, 0:1] + D22 * dq[:, 1:2]
        F_diss = -torch.cat([fric_1, fric_2], dim=-1)
        
        return F_cons + F_diss

# 3. HIGH-ORDER CONTROL BARRIER FUNCTION QP SOLVER
class SafeCBFFilter:
    def __init__(self, env, ph_model, k1=2.0, k2=2.0, delta=0.01):
        self.env = env
        self.model = ph_model
        self.k1 = k1  # Class-K gain for position barrier
        self.k2 = k2  # Class-K gain for velocity barrier
        self.delta = delta
        
        # Online adaptive conformal quantile parameters
        self.cal_residuals = []
        self.N_cal = 200
        self.eps_t = 0.5  # Initial conservative uncertainty bound
        self.beta = 0.05  # Adaptive conformal update step size

    def update_conformal_margin(self, residual_error):
        self.cal_residuals.append(residual_error)
        if len(self.cal_residuals) > self.N_cal:
            self.cal_residuals.pop(0)
            
        # Compute empirical quantile
        q_val = np.quantile(self.cal_residuals, 1.0 - self.delta)
        # Online adaptive step
        indicator = 1.0 if residual_error > self.eps_t else 0.0
        self.eps_t = max(0.05, self.eps_t + self.beta * (indicator - self.delta))

    def solve_qp(self, state, u_nom, mode="cr_ph_cbf"):
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
        else: # Unfiltered baseline
            return u_nom, True

        # Robust High-Order CBF affine inequality: A_cbf * u <= b_cbf
        # b_cbf = k2 * b1 - margin + nominal Lie terms
        b_cbf = self.k2 * b1 - margin + float(A_cbf @ pred_f)
        
        # Solve Quadratic Program: min 0.5 * ||u - u_nom||^2 subject to A_cbf * u <= b_cbf
        # and torque limits [-25, 25]
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

# 4. BENCHMARK VERIFICATION HARNESS (EVALUATING 4 METHODS)
def run_evaluation(num_episodes=20, uncertainty="high"):
    print("="*75)
    print(f"RUNNING RIGOROUS SIMULATION HARNESS | DYNAMICS UNCERTAINTY: {uncertainty.upper()}")
    print("="*75)
    
    env = TwoLinkRobotEnv(uncertainty_level=uncertainty)
    ph_model = PHResNet()
    optimizer = optim.Adam(ph_model.parameters(), lr=1e-3)
    
    # Pre-train PH-ResNet on 200 exploratory nominal transitions
    print("Calibrating Port-Hamiltonian model over initial exploratory phase...")
    for _ in range(200):
        q = np.random.uniform(-np.pi, np.pi, 2)
        dq = np.random.uniform(-2.0, 2.0, 2)
        u = np.random.uniform(-10.0, 10.0, 2)
        
        M0 = env.nominal_inertia(q)
        ddq_meas = np.linalg.inv(M0) @ (u - env.d_nom * dq) + np.random.normal(0, 0.1, 2)
        
        loss = torch.mean((ph_model(torch.FloatTensor(q).unsqueeze(0), 
                                    torch.FloatTensor(dq).unsqueeze(0)) - torch.FloatTensor(ddq_meas))**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    modes = ["unconstrained", "nominal_cbf", "conservative_robust", "cr_ph_cbf"]
    results = {}

    for mode in modes:
        cbf_filter = SafeCBFFilter(env, ph_model)
        episode_successes = []
        episode_violations = []
        episode_max_pen = []
        episode_deadlocks = []
        episode_rewards = []
        
        t0 = time.time()
        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            violations = 0
            max_pen = 0.0
            deadlock_steps = 0
            
            for step in range(250):
                q = state[:2]
                dq = state[2:]
                
                # Nominal PD control directed toward goal
                u_nom = 15.0 * (env.goal_q - q) - 3.0 * dq
                
                # Apply CBF Safety Filter
                if mode == "unconstrained":
                    u_safe = np.clip(u_nom, -25.0, 25.0)
                    feasible = True
                else:
                    u_safe, feasible = cbf_filter.solve_qp(state, u_nom, mode=mode)
                    
                if not feasible:
                    deadlock_steps += 1
                
                # Step physics plant
                next_state, reward, h_val, ddq_true = env.step(u_safe)
                total_reward += reward
                
                # Check barrier penetration (h < 0 means collision)
                if h_val < 0:
                    violations += 1
                    max_pen = max(max_pen, -h_val)
                    
                # Compute residual prediction error & calibrate conformal tracker
                with torch.no_grad():
                    pred_res = ph_model(torch.FloatTensor(q).unsqueeze(0), 
                                        torch.FloatTensor(dq).unsqueeze(0)).numpy().flatten()
                res_err = np.linalg.norm(ddq_true - pred_res)
                cbf_filter.update_conformal_margin(res_err)
                
                state = next_state
                
            dist_final = np.linalg.norm(state[:2] - env.goal_q)
            success = 1.0 if (dist_final < 0.2 and violations == 0) else 0.0
            
            episode_successes.append(success)
            episode_violations.append(violations)
            episode_max_pen.append(max_pen)
            episode_deadlocks.append((deadlock_steps / 250.0) * 100.0)
            episode_rewards.append(total_reward)
            
        results[mode] = {
            "success_rate": np.mean(episode_successes) * 100.0,
            "violations_ep": np.mean(episode_violations),
            "max_violation": np.mean(episode_max_pen),
            "deadlock_rate": np.mean(episode_deadlocks),
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards)
        }
        print(f"Completed mode: {mode:<20} | Elapsed: {time.time()-t0:.2f}s")
        
    # Print publication-ready markdown / LaTeX results table
    print("\n" + "="*85)
    print("EMPIRICAL RESULTS (100% REPRODUCIBLE DATA GENERATED FOR TABLE I)")
    print("="*85)
    print(f"{'Method':<25} | {'Success (%)':<12} | {'Violations':<11} | {'Max Pen (m)':<12} | {'Deadlock (%)':<12} | {'Reward':<12}")
    print("-" * 85)
    for m in modes:
        r = results[m]
        print(f"{m:<25} | {r['success_rate']:<12.1f} | {r['violations_ep']:<11.1f} | {r['max_violation']:<12.3f} | {r['deadlock_rate']:<12.1f} | {r['mean_reward']:<6.1f} ± {r['std_reward']:<5.1f}")
    print("="*85)
    return results

if __name__ == "__main__":
    run_evaluation(num_episodes=20, uncertainty="high")