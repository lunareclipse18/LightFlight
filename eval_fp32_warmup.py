import gym
import numpy as np
import matplotlib.pyplot as plt
from gymfc_nf.envs import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SDF_PATH = "/home/lunareclipse18/LightFlight/models/evoque_v2/model.sdf"

# --- MUST USE THE SAME WRAPPER AS TRAINING ---
class EvoqueWrapper(gym.Wrapper):
    def __init__(self):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(0)
        super().__init__(base_env)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)
        # Tell SB3 the new observation space is 10-dimensional
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self):
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)
        raw_obs = self.env.reset()
        return self._compute_state(raw_obs)

    def _compute_state(self, raw_obs):
        obs = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs.astype(np.float32), -50.0, 50.0)
        current_error = obs[0:3]
        delta_error = current_error - self.prev_error
        delta_error = np.clip(delta_error, -2.0, 2.0)
        self.prev_error = current_error.copy()

        # Expand to 10-neuron state: error + delta_error + prev_action
        state = np.concatenate((current_error, delta_error, self.prev_action))
        return np.float32(state)

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        raw_obs, _, done, info = self.env.step(action)
        state = self._compute_state(raw_obs)

        error_penalty = -np.sum(np.square(state[0:3]))
        control_penalty = -np.sum(np.square(action - 0.5)) * 0.1 
        oscillation_penalty = -np.sum(np.square(action - self.prev_action)) * 0.2

        reward = float(np.clip(
            error_penalty + control_penalty + oscillation_penalty,
            -1000.0, 0.0
        ))
        
        self.prev_action = action.copy()
        return state, reward, done, info

def make_env():
    return EvoqueWrapper()

# Load env — fresh normalizer with warmup
raw_env = DummyVecEnv([make_env])
env = VecNormalize(raw_env, norm_obs=True, norm_reward=False)
env.training = True   # allow stats to accumulate during warmup

# Load the model
model = PPO.load("models/fp32_baseline/best_model", device='cpu')

# ── WARMUP: run 20 episodes to build normalizer stats ──
print("Warming up VecNormalize stats...")
for ep in range(20):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)
        obs, _, dones, _ = env.step(action)
        done = dones[0]
print("Warmup complete.")

# ── FREEZE stats, then do the real eval episode ──
env.training = False

obs = env.reset()
done = False
times, actual, desired = [], [], []
t = 0
inner_env = env.envs[0]

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = np.clip(action, 0.0, 1.0)
    obs, reward, dones, info = env.step(action)
    times.append(t)
    actual.append(inner_env.env.imu_angular_velocity_rpy.copy())
    desired.append(inner_env.env.angular_rate_sp.copy())
    t += inner_env.env.stepsize
    done = dones[0]

env.close()

times   = np.array(times)
actual  = np.array(actual)
desired = np.array(desired)

# Plotting the results
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
labels = ['roll (p)', 'pitch (q)', 'yaw (r)']
for i, ax in enumerate(axes):
    ax.plot(times, desired[:, i], 'k--', label='Desired')
    ax.plot(times, actual[:, i],  'b-',  label='Actual')
    ax.set_ylabel('deg/s')
    ax.set_title(labels[i])
    ax.legend()
    ax.grid(True)

plt.suptitle('Step Response - FP32 LightFlight Policy')
plt.tight_layout()
plt.savefig("fp32_step_response.png")
print("Saved to fp32_step_response.png")