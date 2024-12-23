import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HERCompatiblePusherWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(HERCompatiblePusherWrapper, self).__init__(env)

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float64),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_obs = self.observation(obs)
        reward = self.compute_reward(
            wrapped_obs['achieved_goal'],
            wrapped_obs['desired_goal'],
            info
        )
        return wrapped_obs, reward, terminated, truncated, info

    def observation(self, obs):
        #extracting components from the original observation
        observation = obs[:17]  #joint positions, velocities, and fingertip position
        achieved_goal = obs[17:20]  #object position
        desired_goal = obs[20:23]  #goal position
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0.05).astype(np.float32)  # Binary reward
