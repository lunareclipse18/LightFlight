import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from gymfc_nf.envs import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SDF_PATH   = "/home/lunareclipse18/LightFlight/models/evoque_v2/model.sdf"
MODEL_PATH = "models/fp32_baseline/best_model"
PKL_PATH   = "models/fp32_baseline/best_vec_normalize.pkl"

NUM_EVAL_SEEDS = 30


class EvoqueWrapper(gym.Wrapper):
    def __init__(self, seed=None):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(seed)
        super().__init__(base_env)
        self.prev_error = np.zeros(3, dtype=np.float32)
        # 6-dim obs: error×3 + delta_error×3 — matches training script exactly
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def reset(self):
        self.prev_error = np.zeros(3, dtype=np.float32)
        return self._compute_state(self.env.reset())

    def _compute_state(self, raw_obs):
        obs = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs.astype(np.float32), -500.0, 500.0)
        current_error = obs[0:3]
        delta_error   = np.clip(current_error - self.prev_error, -50.0, 50.0)
        self.prev_error = current_error.copy()
        return np.float32(np.concatenate((current_error, delta_error)))  # 6-dim

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        raw_obs, _, done, info = self.env.step(action)
        state = self._compute_state(raw_obs)
        return state, 0.0, done, info


def make_env(seed=None):
    def _init():
        return EvoqueWrapper(seed=seed)
    return _init


# ── Load env + normalization stats ───────────────────────────────────────────
raw_env = DummyVecEnv([make_env(seed=0)])

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(
        f"No pkl found at {PKL_PATH}. "
        "If you are evaluating the pre-restart model, use eval_fp32_warmup.py instead."
    )

print(f"Loading normalization stats from {PKL_PATH}")
env = VecNormalize.load(PKL_PATH, raw_env)
env.training    = False  # freeze — do not update running stats
env.norm_reward = False  # rewards unused in eval

model = PPO.load(MODEL_PATH, device='cpu')
inner_env = env.envs[0]

# ── Multi-seed eval ───────────────────────────────────────────────────────────
print(f"Running step response eval over {NUM_EVAL_SEEDS} seeds...")
all_rmse          = []
all_times         = []
all_actual        = []
all_desired       = []

for seed in range(NUM_EVAL_SEEDS):
    # Re-seed the inner gymfc env before each episode
    inner_env.env.seed(seed)
    obs  = env.reset()
    done = False
    times, actual, desired = [], [], []
    t = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        times.append(t)
        actual.append(inner_env.env.imu_angular_velocity_rpy.copy())
        desired.append(inner_env.env.angular_rate_sp.copy())
        t   += inner_env.env.stepsize
        done = dones[0]

    actual_arr  = np.array(actual)
    desired_arr = np.array(desired)
    rmse = np.sqrt(np.mean((actual_arr - desired_arr) ** 2, axis=0))
    all_rmse.append(rmse)

    # Keep seed=0 data for the representative plot
    if seed == 0:
        all_times   = np.array(times)
        all_actual  = actual_arr
        all_desired = desired_arr

    print(f"  Seed {seed:2d}  roll={rmse[0]:.2f}  pitch={rmse[1]:.2f}  yaw={rmse[2]:.2f}  avg={rmse.mean():.2f} deg/s")

env.close()

# ── Aggregate stats ──────────────────────────────────────────────────────────
all_rmse = np.array(all_rmse)           # shape (NUM_EVAL_SEEDS, 3)
mean_rmse = all_rmse.mean(axis=0)
std_rmse  = all_rmse.std(axis=0)
avg_rmse  = all_rmse.mean()

print(f"\n{'='*60}")
print(f"Results over {NUM_EVAL_SEEDS} seeds:")
print(f"  Mean RMSE  roll={mean_rmse[0]:.2f}  pitch={mean_rmse[1]:.2f}  yaw={mean_rmse[2]:.2f}  avg={avg_rmse:.2f} deg/s")
print(f"  Std        roll={std_rmse[0]:.2f}   pitch={std_rmse[1]:.2f}   yaw={std_rmse[2]:.2f}")
print(f"\n  NeuroFlight baseline: ~13.8 deg/s avg")
print(f"  {'BEATS' if avg_rmse < 13.8 else 'BELOW'} NeuroFlight baseline")
print(f"{'='*60}")

# ── Plot: step response (seed=0) + per-axis RMSE distribution ────────────────
fig = plt.figure(figsize=(14, 10))
axes_labels = ['roll (p)', 'pitch (q)', 'yaw (r)']

# Top 3 rows: step response traces for seed=0
for i, label in enumerate(axes_labels):
    ax = fig.add_subplot(4, 2, i * 2 + 1)
    ax.plot(all_times, all_desired[:, i], 'k--', label='Desired')
    ax.plot(all_times, all_actual[:, i],  'b-',  label='Actual')
    ax.set_ylabel('deg/s')
    ax.set_title(f"{label}  (seed=0 RMSE={all_rmse[0, i]:.2f} deg/s)")
    ax.legend(fontsize=8)
    ax.grid(True)

# Right column: per-axis RMSE distributions across seeds
for i, label in enumerate(axes_labels):
    ax = fig.add_subplot(4, 2, i * 2 + 2)
    ax.hist(all_rmse[:, i], bins=10, color='steelblue', edgecolor='white')
    ax.axvline(mean_rmse[i], color='red',    linestyle='--', label=f'Mean {mean_rmse[i]:.2f}')
    ax.axvline(13.8,          color='orange', linestyle=':',  label='NF 13.8')
    ax.set_xlabel('RMSE (deg/s)')
    ax.set_title(f"{label} distribution ({NUM_EVAL_SEEDS} seeds)")
    ax.legend(fontsize=8)
    ax.grid(True)

# Bottom row: overall avg RMSE per seed
ax_bottom = fig.add_subplot(4, 1, 4)
seed_avgs = all_rmse.mean(axis=1)
ax_bottom.bar(range(NUM_EVAL_SEEDS), seed_avgs, color='steelblue', edgecolor='white')
ax_bottom.axhline(avg_rmse, color='red',    linestyle='--', linewidth=1.5, label=f'Mean {avg_rmse:.2f}')
ax_bottom.axhline(13.8,      color='orange', linestyle=':',  linewidth=1.5, label='NeuroFlight 13.8')
ax_bottom.set_xlabel('Seed')
ax_bottom.set_ylabel('Avg RMSE (deg/s)')
ax_bottom.set_title(f'Avg RMSE per seed  (overall mean={avg_rmse:.2f} deg/s)')
ax_bottom.legend(fontsize=9)
ax_bottom.grid(True, axis='y')

plt.suptitle(f'FP32 LightFlight — {NUM_EVAL_SEEDS}-seed Eval  (mean avg RMSE {avg_rmse:.2f} deg/s)', fontsize=13)
plt.tight_layout()
plt.savefig("fp32_step_response.png", dpi=150)
print("\nSaved fp32_step_response.png")