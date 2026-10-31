import numpy as np
class TwoLinkRobotEnv:
    
    def __init__(self, uncertainty_level="high"):
        # Control frequency
        self.dt = 0.01  # 100 Hz control loop
        
        # Robot geometry
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
        self.g = 9.81  # Gravity (m/s^2)
        
        # Obstacle Keep-Out Zone (Cartesian End-Effector Space)
        self.obs_center = np.array([1.1, 0.4])
        self.obs_radius = 0.35  # Safety boundary R
        
        # Goal configuration
        self.goal_q = np.array([np.pi/3, -np.pi/4])
        self.reset()

    def reset(self):
        """
        Reset robot to initial configuration.
        
        Returns
        -------
        state : ndarray, shape (4,)
            Initial state [q1, q2, dq1, dq2]
        """
        # Start state [q1, q2, dq1, dq2]
        self.state = np.array([0.05, 0.1, 0.0, 0.0])
        return self.state.copy()

    def forward_kinematics(self, q):
        """
        Compute end-effector position in Cartesian space.
        
        Parameters
        ----------
        q : ndarray, shape (2,)
            Joint angles [q1, q2]
            
        Returns
        -------
        p_ee : ndarray, shape (2,)
            End-effector position [x, y]
        """
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def jacobian(self, q):
        """
        Compute geometric Jacobian matrix J(q).
        
        Maps joint velocities to end-effector velocities: dx/dt = J(q) * dq/dt
        
        Parameters
        ----------
        q : ndarray, shape (2,)
            Joint angles [q1, q2]
            
        Returns
        -------
        J : ndarray, shape (2, 2)
            Jacobian matrix
        """
        J = np.array([
            [-self.l1 * np.sin(q[0]) - self.l2 * np.sin(q[0] + q[1]), -self.l2 * np.sin(q[0] + q[1])],
            [ self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]),  self.l2 * np.cos(q[0] + q[1])]
        ])
        return J

    def nominal_inertia(self, q):
        """
        Compute nominal (assumed) inertia matrix M0(q).
        
        Parameters
        ----------
        q : ndarray, shape (2,)
            Joint angles [q1, q2]
            
        Returns
        -------
        M0 : ndarray, shape (2, 2)
            Nominal inertia matrix
        """
        a1 = 3.0 + 1.5
        a2 = 1.5 * np.cos(q[1])
        a3 = 1.5
        M0 = np.array([[a1 + 2*a2, a3 + a2],
                       [a3 + a2,    a3]])
        return M0

    def true_inertia(self, q):
        """
        Compute true (actual) inertia matrix M(q) with uncertainty.
        
        Parameters
        ----------
        q : ndarray, shape (2,)
            Joint angles [q1, q2]
            
        Returns
        -------
        M : ndarray, shape (2, 2)
            True inertia matrix
        """
        m2 = self.m2_true
        a1 = (self.m1_true + m2) * self.l1**2 + m2 * self.l2**2
        a2 = m2 * self.l1 * self.l2 * np.cos(q[1])
        a3 = m2 * self.l2**2
        M = np.array([[a1 + 2*a2, a3 + a2],
                      [a3 + a2,    a3]])
        return M

    def step(self, u):
        """
        Execute one simulation timestep with applied control torque.
        
        Integrates the manipulator equations of motion:
            M(q) * ddq = u - g(q) - f(dq) + tau_ext
        
        Parameters
        ----------
        u : ndarray, shape (2,)
            Control torque input [tau1, tau2]
            
        Returns
        -------
        state : ndarray, shape (4,)
            Next state [q1, q2, dq1, dq2]
        reward : float
            Negative distance to goal minus control effort
        h : float
            Barrier function value (h < 0 indicates collision)
        ddq : ndarray, shape (2,)
            True joint acceleration (for residual learning)
        """
        q = self.state[:2]
        dq = self.state[2:]
        
        # Compute true non-linear continuous physics
        M = self.true_inertia(q)
        M_inv = np.linalg.inv(M)
        
        # Gravity torques
        g_true = np.array([
            (self.m1_true*0.5 + self.m2_true)*self.g*np.cos(q[0]) + self.m2_true*self.g*np.cos(q[0]+q[1]),
            self.m2_true*self.g*np.cos(q[0]+q[1])
        ])
        
        # Joint friction: stiction + viscous
        fric = self.friction_mult * (self.d_nom * dq + 0.5 * np.sign(dq))
        
        # Wind perturbation on end-effector transformed via J^T
        J = self.jacobian(q)
        tau_wind = J.T @ np.array([self.wind_force, -self.wind_force * 0.5])
        
        # Acceleration: ddq = M^-1 * (u - g - fric + tau_wind)
        ddq = M_inv @ (u - g_true - fric + tau_wind)
        
        # Semi-implicit Euler integration
        dq_new = dq + ddq * self.dt
        q_new = q + dq_new * self.dt
        self.state = np.concatenate([q_new, dq_new])
        
        # Reward: progress toward goal minus control effort
        dist_to_goal = np.linalg.norm(q_new - self.goal_q)
        reward = -dist_to_goal - 0.01 * np.sum(u**2)
        
        # Safety Barrier Metric: h(q) = ||p_ee - p_obs||^2 - R^2
        p_ee = self.forward_kinematics(q_new)
        dist_obs = np.linalg.norm(p_ee - self.obs_center)
        h = dist_obs**2 - self.obs_radius**2
        
        return self.state.copy(), reward, h, ddq
