#Training I2A agents for the Treasure Finder multi agent environment
#importing the dependencies
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
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

from Treasure_FinderEnv import EnvFindTreasure
from Environment_Model import EnvironmentModel
from Imagination_Core import I2A_FindTreasure
from Model_Free_A2C import ModelFreeAgent

# Defining the hyperparameters for both agents
class Hyperparameters:
    def __init__(self, num_episodes, batch_size, replay_memory_size, rollout_len, gamma, lr):
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lr = lr

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

args = get_args()
hyperparameters_agent1 = Hyperparameters(
    num_episodes=args.num_episodes,
    batch_size=args.batch_size,
    replay_memory_size=args.replay_memory_size,
    rollout_len=args.rollout_len,
    gamma=args.gamma,
    lr=args.lr
)
hyperparameters_agent2 = Hyperparameters(
    num_episodes=args.num_episodes,
    batch_size=args.batch_size,
    replay_memory_size=args.replay_memory_size,
    rollout_len=args.rollout_len,
    gamma=args.gamma,
    lr=args.lr
)

# Defining the replay memory for both agents
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
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        return len(self.memory)

# Creating replay memory for both agents
replay_memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)
replay_memory_agent2 = ReplayMemory(hyperparameters_agent2.replay_memory_size)

# Function to select an action using the current policy
def select_action(model, state, output_size, epsilon):
    #if epsilon > 0.95:
    if np.random.rand() < epsilon:
        # Random action
        action = torch.randint(0, output_size[0], (1,))
    else:
        # Use I2A module to produce action
        action_space = torch.tensor([output_size[0]], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # TODO: At moment, this does not return action probs, but concatenated imagined states
            action_probs, _ = model(state, action_space)
            #print("action probe shape:", action_probs.shape)

        m = torch.distributions.Categorical(logits=action_probs)
        action = m.sample()

    return action.item() 

# Main function to train and test the I2A agent

# Create the environment
env = EnvFindTreasure(7)

input_size = (7, 7, 3)  # Update with the appropriate state size attribute
output_size = (5,)
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01


# Instantiate the I2A models and optimizers for both agents
model_agent1 = I2A_FindTreasure(input_size, output_size, args.rollout_len)
model_agent2 = I2A_FindTreasure(input_size, output_size, args.rollout_len)
optimizer_agent1 = optim.Adam(model_agent1.parameters(), lr=args.lr)
optimizer_agent2 = optim.Adam(model_agent2.parameters(), lr=args.lr)

# Initialize the replay memory for both agents
memory_agent1 = ReplayMemory(args.replay_memory_size)
memory_agent2 = ReplayMemory(args.replay_memory_size)

# Main training loop
for episode in range(args.num_episodes):
    state = env.reset()
    state = np.swapaxes(state, 2, 0)
    state = np.expand_dims(state, axis=0)
    print("reset environment:", state.shape)
    dones = False
    episode_reward_agent1 = 0
    episode_reward_agent2 = 0
    max_time_steps = 100  # Maximum number of time steps

    for t in count():
        # Select actions based on the current policies for both agents

        action_agent1 = select_action(model_agent1, state, output_size,epsilon)
        action_agent2 = select_action(model_agent2, state, output_size,epsilon)

        # Execute the actions and store the experiences for both agents
        action_list= [action_agent1, action_agent2]
        reward, done = env.step(action_list)

        
        next_state1 = env.get_global_obs()
        next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
        next_state2 = env.get_global_obs()
        next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state2 = np.expand_dims(next_state2, axis=0)

        if state.shape != (1,3,7,7) and next_state1 != (1,3,7,7) and next_state2 != (1,3,7,7):
            print("ERROR")
            break
        memory_agent1.push(state, action_agent1, reward, next_state1, done)
        memory_agent2.push(state, action_agent2, reward, next_state2, done)

        #TRAIN AGENTS********************************************************************************************


        #if len(memory_agent1) < args.batch_size:
            #return

        if len(memory_agent1) >= args.batch_size:

            states1, actions1, rewards1, next_states1, dones = memory_agent1.sample(args.batch_size)
            states2, actions2, rewards2, next_states2, dones = memory_agent2.sample(args.batch_size)
            #print(len(states1), len(states2))

            for i, state in enumerate(states1):
                #print(state.shape)
                states1[i] = torch.Tensor(state)  # Convert to a PyTorch tensor
            states1 = torch.cat(states1)

            temp_states2 = []
            for state in states2:
                temp_states2.append(torch.Tensor(state))  # Convert to a PyTorch tensor
            states2 = torch.cat(temp_states2)

            next_state_tensor1 = []
            for next_state in next_states1:
                next_state_tensor1.append(torch.Tensor(next_state))

            next_states1 = torch.cat(next_state_tensor1)

            next_state_tensor2 = []
            for next_state in next_states2:
                next_state_tensor2.append(torch.Tensor(next_state))

            next_states2 = torch.cat(next_state_tensor2)

            actions1 = torch.LongTensor(actions1)
            actions2 = torch.LongTensor(actions2)
            rewards1 = torch.Tensor(rewards1)
            rewards2 = torch.Tensor(rewards2)
            dones = torch.Tensor(dones)
            
            #print("state shape befoe feeding to model:", states1.shape, states2.shape)


            # Compute the current Q values for both agents
            # TODO: action prob 
            action_probs_agent1, state_values_agent1 = model_agent1(states1, actions1)
            actions1 = actions1.unsqueeze(-1)
            action_values_agent1 = torch.gather(action_probs_agent1, 1,actions1)

            action_probs_agent2, state_values_agent2 = model_agent2(states2, actions2)
            actions2 = actions2.unsqueeze(-1)
            action_values_agent2 = torch.gather(action_probs_agent2, 1, actions2)

            # Compute the target Q values for both agents
            _, next_state_values_agent1 = model_agent1(next_states1, actions1)
            target_action_values_agent1 = rewards1 + (args.gamma * next_state_values_agent1 * (1 - dones))

            _, next_state_values_agent2 = model_agent2(next_states2, actions2)
            target_action_values_agent2 = rewards2 + (args.gamma * next_state_values_agent2 * (1 - dones))

            # Compute the losses and perform training steps for both agents
            loss_agent1 = (action_values_agent1 - target_action_values_agent1.detach()).pow(2).mean()
            optimizer_agent1.zero_grad()
            loss_agent1.backward()
            optimizer_agent1.step()

            loss_agent2 = (action_values_agent2 - target_action_values_agent2.detach()).pow(2).mean()
            optimizer_agent2.zero_grad()
            loss_agent2.backward()
            optimizer_agent2.step()

        # Update the states and episode rewards for both agents
        state = next_state1
        episode_reward_agent1 += reward
        episode_reward_agent2 += reward

        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        if done or t >= max_time_steps:
            break

    # Check if the episode is finished
    #if done:
    print("Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
        episode, episode_reward_agent1, episode_reward_agent2, t + 1))
    #    break

# Testing the trained agents
print("Testing the trained agents...")
test_episodes = 10
test_rewards_agent1 = []
test_rewards_agent2 = []

for episode in range(test_episodes):
    state = env.reset()
    state = np.swapaxes(state, 2, 0)
    state = np.expand_dims(state, axis=0)
    
    dones = False
    episode_reward_agent1 = 0
    episode_reward_agent2 = 0
    max_time_steps = 100  # Maximum number of time steps

    for t in count():
        action_agent1 = select_action(model_agent1, state, output_size,epsilon)
        action_agent2 = select_action(model_agent2, state, output_size,epsilon)
        # Execute the actions and store the experiences for both agents
        action_list= [action_agent1, action_agent2]
        reward, done = env.step(action_list)
        
        next_state1 = env.get_global_obs()
        next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
        next_state2 = env.get_global_obs()
        next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state2 = np.expand_dims(next_state2, axis=0)
        
        memory_agent1.push(state, action_agent1, reward, next_state1, done)
        memory_agent2.push(state, action_agent2, reward, next_state2, done)
        
        rewards1 = torch.Tensor(rewards1)
        rewards2 = torch.Tensor(rewards2)
        
        #dones = torch.Tensor(dones)
        episode_reward_agent1 += reward
        episode_reward_agent2 += reward
        state = next_state1
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
            # Check if the episode is finished
        if done or t >= max_time_steps:
            break
    print("Test Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
        episode, episode_reward_agent1, episode_reward_agent2, t + 1))
    test_rewards_agent1.append(episode_reward_agent1)
    test_rewards_agent2.append(episode_reward_agent2)


print("Average test reward Agent 1: {:.2f}".format(sum(test_rewards_agent1) / test_episodes))
print("Average test reward Agent 2: {:.2f}".format(sum(test_rewards_agent2) / test_episodes))

