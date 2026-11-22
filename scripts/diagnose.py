#!/usr/bin/env python3
"""Quick diagnostic to understand the simulation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crphcbf.environments import TwoLinkRobotEnv
import numpy as np

env = TwoLinkRobotEnv(uncertainty_level="high")

print("=" * 60)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 60)

# Initial state
state = env.reset()
q_init = state[:2]
p_init = env.forward_kinematics(q_init)

print(f"\n1. Initial Configuration:")
print(f"   Joint angles: q = {q_init}")
print(f"   End-effector position: p = {p_init}")
print(f"   Distance from origin: {np.linalg.norm(p_init):.3f} m")

print(f"\n2. Goal Configuration:")
print(f"   Goal angles: q_goal = {env.goal_q}")
q_goal = env.goal_q
p_goal = env.forward_kinematics(q_goal)
print(f"   Goal end-effector: p_goal = {p_goal}")
print(f"   Initial distance to goal: {np.linalg.norm(q_init - q_goal):.3f} rad")

print(f"\n3. Obstacle (Keep-Out Zone):")
print(f"   Center: {env.obs_center}")
print(f"   Radius: {env.obs_radius} m")
print(f"   Distance from init to obstacle: {np.linalg.norm(p_init - env.obs_center):.3f} m")
print(f"   Distance from goal to obstacle: {np.linalg.norm(p_goal - env.obs_center):.3f} m")

# Check barrier function
h_init = np.linalg.norm(p_init - env.obs_center)**2 - env.obs_radius**2
h_goal = np.linalg.norm(p_goal - env.obs_center)**2 - env.obs_radius**2

print(f"\n4. Barrier Function h(q):")
print(f"   h(q_init) = {h_init:.3f} (safe if > 0)")
print(f"   h(q_goal) = {h_goal:.3f} (safe if > 0)")

if h_goal < 0:
    print(f"   ⚠️  WARNING: Goal is INSIDE the obstacle! Task is impossible!")
elif h_goal < 0.01:
    print(f"   ⚠️  WARNING: Goal is very close to obstacle boundary!")
    
print(f"\n5. Dynamics Uncertainty:")
print(f"   Nominal payload: {env.m2_nom:.1f} kg")
print(f"   True payload: {env.m2_true:.1f} kg")
print(f"   Payload shift: +{(env.m2_true - env.m2_nom)/env.m2_nom * 100:.1f}%")
print(f"   Friction multiplier: {env.friction_mult}x")
print(f"   Wind force: {env.wind_force} N")

# Run one step to see dynamics
print(f"\n6. Sample Dynamics Test:")
u_test = np.array([5.0, 5.0])
state_new, reward, h_val, ddq = env.step(u_test)
print(f"   Applied torque: u = {u_test}")
print(f"   Resulting acceleration: ddq = {ddq}")
print(f"   New state: {state_new}")
print(f"   Reward: {reward:.2f}")
print(f"   Barrier value: h = {h_val:.3f}")

print("\n" + "=" * 60)
