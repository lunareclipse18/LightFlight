import gym
import numpy as np
import matplotlib.pyplot as plt
from gymfc_nf.envs import *

SDF_PATH = "/home/lunareclipse18/LightFlight/models/evoque_v2/model.sdf"

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

class EvoqueWrapper(gym.Wrapper):
    def __init__(self):
        base_env = gym.make("gymfc_nf-step-v1")
        base_env.set_aircraft_model(SDF_PATH)
        base_env.seed(0)
        super().__init__(base_env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        obs, _, done, info = self.env.step(action)
        return obs, 0.0, done, info


pid = CascadePID()
env = EvoqueWrapper()
obs = env.reset()
pid.reset()

dt   = env.env.stepsize
done = False
times, actual, desired = [], [], []
t = 0

while not done:
    setpoint = env.env.angular_rate_sp.copy()
    error    = setpoint - env.env.imu_angular_velocity_rpy
    action   = pid.step(error, dt, setpoint)
    obs, _, done, _ = env.step(action)

    times.append(t)
    actual.append(env.env.imu_angular_velocity_rpy.copy())
    desired.append(env.env.angular_rate_sp.copy())
    t += dt

env.close()

times   = np.array(times)
actual  = np.array(actual)
desired = np.array(desired)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
labels = ['roll (p)', 'pitch (q)', 'yaw (r)']
for i, ax in enumerate(axes):
    ax.plot(times, desired[:, i], 'k--', label='Desired')
    ax.plot(times, actual[:, i],  'r-',  label='PID Actual')
    ax.set_ylabel('deg/s')
    ax.set_title(labels[i])
    ax.legend()
    ax.grid(True)

plt.suptitle('Step Response - Cascade PID Baseline')
plt.tight_layout()
plt.savefig("pid_step_response.png")
print("Saved to pid_step_response.png")