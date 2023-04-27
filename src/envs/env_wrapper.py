import gymnasium as gym
import numpy as np

#defining the environment class
class EnvWrapper(gym.Env):
    def __init__(self,env_name='MountainCar-v0'):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def reset(self):
        state = self.env.reset()
        print("Original state:", state)
        return np.array(state[0])


    def step(self, action):
            next_state, reward, done, info = self.env.step(action)[0:4]
            return np.array(next_state), reward, done, info # Return the full next_state

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

