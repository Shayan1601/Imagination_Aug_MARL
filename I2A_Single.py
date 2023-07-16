import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.distributions import Categorical
import argparse
import sys
import gymnasium as gym
from gym import wrappers
from torch.utils.data import DataLoader
from itertools import count
from collections import namedtuple

from gym.spaces import Discrete


#defining the environment class
class MountainCarWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def reset(self):
        state = self.env.reset()
        print("Original state:", state)
        return np.array(state[0])


    def step(self, action):
            next_state, reward, done, _ = self.env.step(action)[0:4]
            return np.array(next_state), reward, done  # Return the full next_state

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


#building the environment model
class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnvironmentModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + 1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.state_head = nn.Linear(32, state_dim)

    def forward(self, state, action):
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).repeat(state.size(0), 1)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.state_head(x)
        return next_state

class I2A_MountainCar(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len, hidden_dim=256):
        super(I2A_MountainCar, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_len = rollout_len

        # Define the environment model
        self.env_model = EnvironmentModel(state_dim, action_dim)

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(state_dim * rollout_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_space):
        # Rollout the environment model
        imagined_states = []
        for _ in range(self.rollout_len):
            action = action_space.sample()
            state = self.env_model(state, action)
            imagined_states.append(state)
        imagined_states = torch.cat(imagined_states, dim=-1)

        # Encode the imagined states
        x = self.imagination(imagined_states)

        # Compute the action probabilities
        action_probs = self.policy_head(x)

        # Compute the state values
        state_values = self.value_head(x)

        return action_probs, state_values



#defining the hyperparameters
def get_args():
    parser = argparse.ArgumentParser(description="Imagination-Augmented Agents for Deep Reinforcement Learning")

    # Training settings
    parser.add_argument("--num_episodes", type=int, default=30, help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("--rollout_len", type=int, default=5, help="Length of the rollout for imagination")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")

    if sys.argv[0].endswith("ipykernel_launcher.py"):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args


import random
from collections import namedtuple
#defining the replayMemory for storing experience trajectories
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# Define the namedtuple to store experiences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# function to select an action using the current policy
def select_action(model, state, action_space):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs, _ = model(state, action_space)
    m = torch.distributions.Categorical(action_probs)
    return m.sample().item()




# Main function to train and test the I2A agent
def main(args):
    # Create the environment
    env = MountainCarWrapper()
    state = env.reset()


    # Instantiate the I2A model and optimizer
    model = I2A_MountainCar(env.observation_space.shape[0], env.action_space.n, args.rollout_len)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the replay memory
    memory = ReplayMemory(args.replay_memory_size)

    # Main training loop
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0

        for t in count():
            # Select an action based on the current policy
            action = select_action(model, state, env.action_space)


            # Execute the action and store the experience
            next_state, reward, done= env.step(action)[0:3]
            memory.push(state, action, reward, next_state, done)



            # Update the state and episode reward
            state = next_state
            episode_reward += reward

            # If enough experiences are collected, perform a training step
            if len(memory) >= args.batch_size:
                experiences = memory.sample(args.batch_size)
                batch = Experience(*zip(*experiences))

               
                # Prepare the data for training
                states = torch.tensor(np.array(batch.state), dtype=torch.float32)
                actions = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
                dones = torch.tensor(np.array(batch.done), dtype=torch.float32).unsqueeze(1)


                # Compute the current Q values
                action_probs, state_values = model(states, env.action_space)
                action_values = action_probs.gather(1, actions)

                # Compute the target Q values
                _, next_state_values = model(next_states, env.action_space)

                target_action_values = rewards + (args.gamma * next_state_values * (1 - dones))

                # Compute the loss and perform a training step
                loss = (action_values - target_action_values.detach()).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Check if the episode is finished
            if done:
                print("Episode: {}, Reward: {}, Timesteps: {}".format(episode, episode_reward, t + 1))
                break

    # Testing the trained agent
    print("Testing the trained agent...")
    test_episodes = 10
    test_rewards = []

    for episode in range(test_episodes):
        state = env.reset()
        episode_reward = 0

        for t in count():
            action = select_action(model, state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                print("Test Episode: {}, Reward: {}, Timesteps: {}".format(episode, episode_reward, t + 1))
                test_rewards.append(episode_reward)
                break

    print("Average test reward: {:.2f}".format(sum(test_rewards) / test_episodes))
    env.close()
if __name__ == "__main__":

    args = get_args()
    main(args)
