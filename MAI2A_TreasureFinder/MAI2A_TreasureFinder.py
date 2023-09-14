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

from Treasure_Finder_gymformat import TreasureFinderEnv
from Environment_Model import EnvironmentModel
from Imagination_Core import I2A_FindTreasure
from Model_Free_A2C import ModelFreeAgent
from config1 import hyperparameters_agent1, hyperparameters_agent2

# Defining the hyperparameters for both agents

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
            
            action_probs, _ = model(state, action_space)
            #print("action probe shape:", action_probs.shape)

        m = torch.distributions.Categorical(logits=action_probs)
        action = m.sample()

    return action.item() 

# Main function to train and test the I2A agent

# Create the environment
env = TreasureFinderEnv(7)

input_size = (7, 7, 3)  # Update with the appropriate state size attribute
output_size = (5,)
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01


# Instantiate the I2A models and optimizers for both agents
model_agent1 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent1.rollout_len)
model_agent2 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent2.rollout_len)
optimizer_agent1 = optim.Adam(model_agent1.parameters(), lr=hyperparameters_agent1.lr)
optimizer_agent2 = optim.Adam(model_agent2.parameters(), lr=hyperparameters_agent2.lr)

# Initialize the replay memory for both agents
memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)
memory_agent2 = ReplayMemory(hyperparameters_agent2.replay_memory_size)

# Main training loop


mean_rewards_agent1 = []  # To store mean rewards for Agent 1
mean_rewards_agent2 = []  # To store mean rewards for Agent 2


for episode in range(hyperparameters_agent1.num_episodes):
    state = env.reset()
    state = np.swapaxes(state, 2, 0)
    state = np.expand_dims(state, axis=0)
    #print("reset environment:", state.shape) #(1,3,7,7)
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
        next_state1,reward, done = env.step(action_list)

        
        
        #print("state shape right after GET_GlobaL",next_state1.shape) --> (7,7,3)
        next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
        next_state2 = next_state1
        #next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        #next_state2 = np.expand_dims(next_state2, axis=0)
        #print("next state shape right after GET_GlobaL",next_state1.shape) (1,3,7,7)
        
        

        if state.shape != (1,3,7,7) and next_state1 != (1,3,7,7) and next_state2 != (1,3,7,7):
            print("ERROR")
            break
        memory_agent1.push(state, action_agent1, reward, next_state1, done)
        memory_agent2.push(state, action_agent2, reward, next_state2, done)

        #TRAIN AGENTS********************************************************************************************


        #if len(memory_agent1) < args.batch_size:
            #return

        if len(memory_agent1) >= hyperparameters_agent1.batch_size:

            states1, actions1, rewards1, next_states1, dones = memory_agent1.sample(hyperparameters_agent1.batch_size)
            states2, actions2, rewards2, next_states2, dones = memory_agent2.sample(hyperparameters_agent2.batch_size)
            #print("state shape right after SAMPLING",states1) #-> 60
            #print(states1.shape)
            Trajectory = [states1, actions1, rewards1, next_states1]

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
            
            #print("state shape befoe feeding to model:", states1.shape, states2.shape) --> ([60,3,7,7])


            # Compute the current Q values for both agents
            
            action_probs_agent1, state_values_agent1 = model_agent1(states1, actions1)
            #print(action_probs_agent1)
            actions1 = actions1.unsqueeze(-1)
            action_values_agent1 = torch.gather(action_probs_agent1, 1,actions1)
            #print("action value:",action_values_agent1)

            action_probs_agent2, state_values_agent2 = model_agent2(states2, actions2)
            actions2 = actions2.unsqueeze(-1)
            action_values_agent2 = torch.gather(action_probs_agent2, 1, actions2)

            # Compute the target Q values for both agents
            _, next_state_values_agent1 = model_agent1(next_states1, actions1)
            target_action_values_agent1 = rewards1 + (hyperparameters_agent1.gamma * next_state_values_agent1 * (1 - dones))
            #print("Target value:",target_action_values_agent1)

            _, next_state_values_agent2 = model_agent2(next_states2, actions2)
            target_action_values_agent2 = rewards2 + (hyperparameters_agent2.gamma * next_state_values_agent2 * (1 - dones))
            


            # Compute the losses and perform training steps for both agents
            loss_agent1 = (action_values_agent1 - target_action_values_agent1.detach()).pow(2).mean()
            optimizer_agent1.zero_grad()
            loss_agent1.backward()
            optimizer_agent1.step()

            loss_agent2 = (action_values_agent2 - target_action_values_agent2.detach()).pow(2).mean()
            optimizer_agent2.zero_grad()
            loss_agent2.backward()
            optimizer_agent2.step()
            #Compute the current Q values for both agents
            
            # #new mthods PPO
            # # Define PPO-specific hyperparameters
            # epsilon_clip = 0.2
            # ppo_epochs = 10
            # batch_size = 60
            
            # # Collect trajectories (assuming you have a function to collect trajectories)
            # #trajectories = Trajectory

            # # Loop through trajectories and compute advantages
            # advantages1 = []
            # advantages2 = []

            # for t in range(len(states1)):
            #     state = states1[t]
            #     action = actions1[t]
            #     reward = rewards1[t]
            #     next_state = next_states1[t]

            #     # Compute Q-values and state-values for both agents
            #     q_value_agent1, _ = model_agent1(state, action)
            #     v_value_agent1, _ = model_agent1(state)

            #     q_value_agent2, _ = model_agent2(state, action)
            #     v_value_agent2, _ = model_agent2(state)

            #     # Calculate advantages
            #     advantage1 = q_value_agent1 - v_value_agent1
            #     advantage2 = q_value_agent2 - v_value_agent2

            #     # Append to advantages lists
            #     advantages1.append(advantage1)
            #     advantages2.append(advantage2)

            # # Convert advantages to tensors
            # advantages1 = torch.cat(advantages1)
            # advantages2 = torch.cat(advantages2)


            # # Initialize optimizers for both agents
            # optimizer_agent1 = optim.Adam(model_agent1.parameters(), lr=hyperparameters_agent1.lr)
            # optimizer_agent2 = optim.Adam(model_agent2.parameters(), lr=hyperparameters_agent2.lr)
            
            # # Initialize old action probabilities
            # action_probs_old_agent1 = torch.zeros(batch_size, 5) 
            # action_probs_old_agent2 = torch.zeros(batch_size, 5) 


            # for _ in range(ppo_epochs):
            #     # Compute the current action probabilities and state values for both agents
            #     action_probs_agent1, state_values_agent1 = model_agent1(states1, actions1)
            #     action_probs_agent2, state_values_agent2 = model_agent2(states2, actions2)

            #     # Compute the ratio of new and old action probabilities
            #     ratio_agent1 = torch.exp(action_probs_agent1 - action_probs_old_agent1)
            #     ratio_agent2 = torch.exp(action_probs_agent2 - action_probs_old_agent2)

            #     # Compute surrogate losses for both agents
            #     surrogate1 = ratio_agent1 * advantages1
            #     surrogate2 = ratio_agent2 * advantages2

            #     # Calculate clipped surrogate losses
            #     clipped_surrogate1 = torch.clamp(ratio_agent1, 1 - epsilon_clip, 1 + epsilon_clip) * advantages1
            #     clipped_surrogate2 = torch.clamp(ratio_agent2, 1 - epsilon_clip, 1 + epsilon_clip) * advantages2

            #     # Choose the minimum of the clipped and unclipped surrogate losses
            #     surrogate_loss_agent1 = -torch.min(surrogate1, clipped_surrogate1).mean()
            #     surrogate_loss_agent2 = -torch.min(surrogate2, clipped_surrogate2).mean()

            #     # Compute the critic loss for both agents
            #     critic_loss_agent1 = F.mse_loss(state_values_agent1, target_state_values1)
            #     critic_loss_agent2 = F.mse_loss(state_values_agent2, target_state_values2)

            #     # Compute the total loss for both agents
            #     loss_agent1 = surrogate_loss_agent1 + critic_loss_agent1
            #     loss_agent2 = surrogate_loss_agent2 + critic_loss_agent2

            #     # Perform optimization steps for both agents
            #     optimizer_agent1.zero_grad()
            #     loss_agent1.backward()
            #     optimizer_agent1.step()

            #     optimizer_agent2.zero_grad()
            #     loss_agent2.backward()
            #     optimizer_agent2.step()

            # # Update the old action probabilities for the next iteration
            # action_probs_old_agent1 = action_probs_agent1.detach()
            # action_probs_old_agent2 = action_probs_agent2.detach()


        # Update the states and episode rewards for both agents
        state = next_state1
        #print("state shape bad az network update", state.shape)--> (1,3,7,7)
        episode_reward_agent1 += reward
        episode_reward_agent2 += reward

        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        if done or t >= max_time_steps:
            break

    # Check if the episode is finished
    #if done:
    #print("Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
        #episode, episode_reward_agent1, episode_reward_agent2, t + 1))
    # Calculate mean rewards for Agent 1 and Agent 2
    mean_rewards_agent1.append(episode_reward_agent1)
    mean_rewards_agent2.append(episode_reward_agent2)

    if (episode + 1) % 100 == 0:
        mean_reward_agent1 = sum(mean_rewards_agent1[-100:]) / min(100, len(mean_rewards_agent1))
        mean_reward_agent2 = sum(mean_rewards_agent2[-100:]) / min(100, len(mean_rewards_agent2))
        print("Episode: {}, Mean Reward Agent 1 (last 100 episodes): {:.2f}".format(episode, mean_reward_agent1))
        print("Mean Reward Agent 2 (last 100 episodes): {:.2f}".format( mean_reward_agent2))
# Print final mean rewards after training
final_mean_reward_agent1 = sum(mean_rewards_agent1) / len(mean_rewards_agent1)
final_mean_reward_agent2 = sum(mean_rewards_agent2) / len(mean_rewards_agent2)
print("Final Mean Reward Agent 1: {:.2f}".format(final_mean_reward_agent1))
print("Final Mean Reward Agent 2: {:.2f}".format(final_mean_reward_agent2))        



# # Testing the trained agents
# print("Testing the trained agents...")
# test_episodes = 1000
# test_rewards_agent1 = []
# test_rewards_agent2 = []

# for episode in range(test_episodes):
#     state = env.reset()
#     state = np.swapaxes(state, 2, 0)
#     state = np.expand_dims(state, axis=0)
    
#     dones = False
#     episode_reward_agent1 = 0
#     episode_reward_agent2 = 0
#     max_time_steps = 100  # Maximum number of time steps

#     for t in count():
#         action_agent1 = select_action(model_agent1, state, output_size,epsilon)
#         action_agent2 = select_action(model_agent2, state, output_size,epsilon)
#         # Execute the actions and store the experiences for both agents
#         action_list= [action_agent1, action_agent2]
#         reward, done = env.step(action_list)
        
#         next_state1 = env.get_global_obs()
#         next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
#         next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
#         next_state2 = env.get_global_obs()
#         next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
#         next_state2 = np.expand_dims(next_state2, axis=0)
        
#         memory_agent1.push(state, action_agent1, reward, next_state1, done)
#         memory_agent2.push(state, action_agent2, reward, next_state2, done)
        
#         rewards1 = torch.Tensor(rewards1)
#         rewards2 = torch.Tensor(rewards2)
        
#         #dones = torch.Tensor(dones)
#         episode_reward_agent1 += reward
#         episode_reward_agent2 += reward
#         state = next_state1
#         epsilon = max(epsilon * epsilon_decay, min_epsilon)
#             # Check if the episode is finished
#         if done or t >= max_time_steps:
#             break
#     print("Test Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
#         episode, episode_reward_agent1, episode_reward_agent2, t + 1))
#     test_rewards_agent1.append(episode_reward_agent1)
#     test_rewards_agent2.append(episode_reward_agent2)


# print("Average test reward Agent 1: {:.2f}".format(sum(test_rewards_agent1) / test_episodes))
# print("Average test reward Agent 2: {:.2f}".format(sum(test_rewards_agent2) / test_episodes))

