import gym
import numpy as np
from gymfc_nf.envs import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SDF_PATH = "/home/lunareclipse18/LightFlight/models/evoque_v2/model.sdf"

# ── Paste your confirmed CascadePID class here (the last working version) ──

class CascadePID:
    def __init__(self):
        self.kp = np.array([0.018, 0.045, 0.000])  # pitch kp 2.5x higher
        self.ki = np.array([0.001, 0.004, 0.000])   # pitch ki 4x higher
        self.kd = np.array([0.0004, 0.0008, 0.000])
        self.integral       = np.zeros(3)
        self.prev_error     = np.zeros(3)
        self.filtered_deriv = np.zeros(3)
        self.prev_sp        = np.zeros(3)
        self.alpha          = 0.08

    def reset(self):
        self.integral       = np.zeros(3)
        self.prev_error     = np.zeros(3)
        self.filtered_deriv = np.zeros(3)
        self.prev_sp        = np.zeros(3)

    def step(self, error, dt, setpoint):
        if not np.allclose(setpoint, self.prev_sp, atol=1.0):
            self.integral = np.zeros(3)
        self.prev_sp = setpoint.copy()

        self.integral += error * dt
        self.integral  = np.clip(self.integral, -5.0, 5.0)

        raw_deriv           = (error - self.prev_error) / dt
        self.filtered_deriv = self.alpha*raw_deriv + (1-self.alpha)*self.filtered_deriv
        self.prev_error     = error.copy()

        roll  = self.kp[0]*error[0] + self.ki[0]*self.integral[0] + self.kd[0]*self.filtered_deriv[0]
        pitch = self.kp[1]*error[1] + self.ki[1]*self.integral[1] + self.kd[1]*self.filtered_deriv[1]
        # Yaw zeroed — torque-based axis requires separate tuning
        
        m0 = 0.5 - roll + pitch
        m1 = 0.5 - roll - pitch
        m2 = 0.5 + roll + pitch
        m3 = 0.5 + roll - pitch

        return np.clip([m0, m1, m2, m3], 0.0, 1.0).astype(np.float32)

class EvoqueWrapperRaw(gym.Wrapper):
    def __init__(self):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(0)
        super().__init__(base_env)
    def reset(self): return self.env.reset()
    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        obs, _, done, info = self.env.step(action)
        return obs, 0.0, done, info

class EvoqueWrapperNN(gym.Wrapper):
    def __init__(self):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(0)
        super().__init__(base_env)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_error  = np.zeros(3, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    def reset(self):
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_error  = np.zeros(3, dtype=np.float32)
        return self._compute_state(self.env.reset())
    def _compute_state(self, raw_obs):
        obs = np.clip(np.nan_to_num(raw_obs, nan=0.0), -50.0, 50.0).astype(np.float32)
        current_error = obs[0:3]
        delta_error   = np.clip(current_error - self.prev_error, -2.0, 2.0)
        self.prev_error = current_error.copy()
        return np.concatenate((current_error, delta_error, self.prev_action))
    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        raw_obs, _, done, info = self.env.step(action)
        self.prev_action = action.copy()
        return self._compute_state(raw_obs), 0.0, done, info

def compute_metrics(actual, desired, dt):
    error = desired - actual
    ise   = np.sum(error**2) * dt
    iae   = np.sum(np.abs(error)) * dt
    itae  = np.sum(np.arange(len(error)) * dt * np.abs(error)) * dt
    return ise, iae, itae

# ── Run PID episode ──────────────────────────────────────────────────────────
print("Running PID episode...")
pid = CascadePID()
env = EvoqueWrapperRaw()
env.reset(); pid.reset()
dt = env.env.stepsize
pid_actual, pid_desired = [], []

done = False
steps = 0
while not done:
    sp     = env.env.angular_rate_sp.copy()
    imu    = env.env.imu_angular_velocity_rpy.copy()
    action = pid.step(sp - imu, dt, sp)
    _, _, done, _ = env.step(action)
    pid_actual.append(imu)
    pid_desired.append(sp)
    steps += 1
    if steps % 500 == 0:
        print(f"  PID step {steps}...")
env.close()

pid_actual  = np.array(pid_actual)
pid_desired = np.array(pid_desired)

# ── Run NN episode ───────────────────────────────────────────────────────────
print("Running NN episode...")
raw_env = DummyVecEnv([lambda: EvoqueWrapperNN()])
nn_env  = VecNormalize(raw_env, norm_obs=True, norm_reward=False)
nn_env.training = True
model = PPO.load("models/fp32_baseline/best_model", device='cpu')

# Warmup
print("Running NN warmup (5 episodes)...")
for ep in range(5):
    obs = nn_env.reset(); done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = nn_env.step(np.clip(action, 0.0, 1.0))
        done = dones[0]
        steps += 1
    print(f"  Warmup episode {ep+1}/5 done ({steps} steps)")

print("Warmup complete. Running eval episode...")

nn_env.training = False
obs = nn_env.reset()
done = False
nn_actual, nn_desired = [], []
inner = nn_env.envs[0]

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, _ = nn_env.step(np.clip(action, 0.0, 1.0))
    nn_actual.append(inner.env.env.imu_angular_velocity_rpy.copy())
    nn_desired.append(inner.env.env.angular_rate_sp.copy())
    done = dones[0]
nn_env.close()

nn_actual  = np.array(nn_actual)
nn_desired = np.array(nn_desired)

# ── Print comparison table ───────────────────────────────────────────────────
axes = ['Roll (p)', 'Pitch (q)', 'Yaw (r)']
print(f"\n{'Axis':<12} {'Metric':<8} {'PID':>12} {'NN':>12} {'Winner':>10}")
print("-" * 58)

for i, ax in enumerate(axes):
    for metric, idx in [('ISE', 0), ('IAE', 1), ('ITAE', 2)]:
        pid_m = compute_metrics(pid_actual[:,i], pid_desired[:,i], dt)[idx]
        nn_m  = compute_metrics(nn_actual[:,i],  nn_desired[:,i],  dt)[idx]
        winner = 'NN' if nn_m < pid_m else 'PID'
        print(f"{ax:<12} {metric:<8} {pid_m:>12.2f} {nn_m:>12.2f} {winner:>10}")
    print()

print("\n── RMSE (deg/s) ──────────────────────────")
for i, ax in enumerate(axes):
    pid_rmse = np.sqrt(np.mean((pid_actual[:,i] - pid_desired[:,i])**2))
    nn_rmse  = np.sqrt(np.mean((nn_actual[:,i]  - nn_desired[:,i])**2))
    print(f"{ax:<12}  PID: {pid_rmse:6.2f}   NN: {nn_rmse:6.2f}")

print("\n── NN Step Response RMSE vs NeuroFlight ──")
print("NeuroFlight published: ~13.8 deg/s avg (Koch et al. 2019)")
nn_avg_rmse = np.mean([
    np.sqrt(np.mean((nn_actual[:,i] - nn_desired[:,i])**2))
    for i in range(3)
])
print(f"Your FP32 baseline:     {nn_avg_rmse:.2f} deg/s avg")
if nn_avg_rmse < 13.8:
    print("Result: MATCHES or EXCEEDS NeuroFlight baseline ✓")
else:
    print("Result: Below NeuroFlight baseline — consider more training")