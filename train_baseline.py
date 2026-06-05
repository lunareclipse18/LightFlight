import gym
import torch.nn as nn
import numpy as np
import os
from gymfc_nf.envs import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Callable


SDF_PATH = "/home/lunareclipse18/LightFlight/models/evoque_v2/model.sdf"


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
    return func


class EvoqueWrapper(gym.Wrapper):
    def __init__(self, seed=None):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(seed)
        super().__init__(base_env)
        self.prev_action       = np.zeros(4, dtype=np.float32)
        self.prev_error        = np.zeros(3, dtype=np.float32)
        self._default_max_rate = None
        self.error_weights = np.array([2.0, 1.5, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    

    def set_max_rate(self, fraction: float):
        inner = self.env
        while inner is not None:
            if hasattr(inner, 'max_rate'):
                if self._default_max_rate is None:
                    self._default_max_rate = inner.max_rate
                inner.max_rate = self._default_max_rate * fraction
                return
            inner = getattr(inner, 'env', None)

    def reset(self):
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_error  = np.zeros(3, dtype=np.float32)
        return self._compute_state(self.env.reset())

    def _compute_state(self, raw_obs):
        obs = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs.astype(np.float32), -500.0, 500.0)
        current_error = obs[0:3]
        delta_error   = np.clip(current_error - self.prev_error, -50.0, 50.0)
        self.prev_error = current_error.copy()
        return np.float32(np.concatenate((current_error, delta_error)))

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        raw_obs, _, done, info = self.env.step(action)

        if np.isnan(raw_obs).any() or np.isinf(raw_obs).any():
            return np.zeros(6, dtype=np.float32), -10000.0, True, info

        state         = self._compute_state(raw_obs)
        current_error = state[0:3]

        

        error_penalty = -(
        np.sum(self.error_weights * np.square(current_error)) +
        np.sum(self.error_weights * np.abs(current_error)) * 0.3
        )
        
        oscillation_penalty = -np.sum(np.square(action - self.prev_action)) * 0.2
        reward = float(np.clip(
            error_penalty + oscillation_penalty,
            -10000.0, 0.0
        ))

        self.prev_action = action.copy()
        return state, reward, done, info


def make_env(seed=None):
    def _init():
        return EvoqueWrapper(seed=seed)
    return _init


class CurriculumCallback(BaseCallback):
    """
    Linearly ramps max_rate from start_fraction to 1.0 over ramp_steps,
    then holds at full difficulty for the rest of training.
    """
    def __init__(self, ramp_steps: int = 3_000_000, start_fraction: float = 0.15,
                 verbose: int = 0):
        super().__init__(verbose)
        self.ramp_steps     = ramp_steps
        self.start_fraction = start_fraction

    def _on_step(self) -> bool:
        if self.num_timesteps % 4096 == 0:
            progress = min(self.num_timesteps / self.ramp_steps, 1.0)
            fraction = self.start_fraction + progress * (1.0 - self.start_fraction)
            venv = getattr(self.training_env, 'venv', self.training_env)
            for env in venv.envs:
                wrapper = env
                while wrapper is not None:
                    if isinstance(wrapper, EvoqueWrapper):
                        wrapper.set_max_rate(fraction)
                        break
                    wrapper = getattr(wrapper, 'gym_env', None) or getattr(wrapper, 'env', None)
        return True


class SyncEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if isinstance(self.training_env, VecNormalize) and \
               isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = self.training_env.obs_rms
                self.eval_env.ret_rms = self.training_env.ret_rms

        prev_best = self.best_mean_reward
        result    = super()._on_step()

        if (self.eval_freq > 0
                and self.n_calls % self.eval_freq == 0
                and self.best_mean_reward > prev_best):
            pkl_path = os.path.join(self.best_model_save_path, "best_vec_normalize.pkl")
            self.training_env.save(pkl_path)

        return result


def main():
    env = DummyVecEnv([make_env(seed=None)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # 5 diverse seeds — best_model selection based on mean over varied setpoints,
    # not just one potentially easy or hard episode
    eval_env = DummyVecEnv([make_env(seed=i) for i in range(5)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=cosine_schedule(3e-4),
        n_steps=4096,
        batch_size=512,
        gamma=0.99,
        ent_coef=0.0005,
        max_grad_norm=0.5,
        target_kl=0.02,
        normalize_advantage=True,
        verbose=1,
        device='cpu',
        tensorboard_log="./lwn_flight_tensorboard/"
    )

    eval_callback = SyncEvalCallback(
        eval_env,
        best_model_save_path="./models/fp32_baseline/",
        log_path="./logs/",
        eval_freq=50_000,
        n_eval_episodes=5,      # one episode per eval env seed
        deterministic=True,
        render=False
    )

    curriculum_callback = CurriculumCallback(
        ramp_steps=3_000_000,   # longer ramp — full difficulty at 3M not 2M
        start_fraction=0.15     # start even easier
    )

    print("Starting FP32 baseline training with curriculum...")
    model.learn(
        total_timesteps=6_000_000,
        callback=[eval_callback, curriculum_callback]
    )
    model.save("models/fp32_baseline/final_fp32_model")
    env.save("models/fp32_baseline/vec_normalize.pkl")
    print("Training complete.")


if __name__ == "__main__":
    main()
