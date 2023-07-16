
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

#defining the environment
class EnvFindTreasure(object):
    def __init__(self, map_size):
        self.map_size = map_size
        if map_size<7:
            self.map_size = 7

        self.half_pos = int((self.map_size - 1)/2)

        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0, i] = 1
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.map_size - 1] = 1
            self.occupancy[self.map_size - 1, i] = 1
            self.occupancy[self.half_pos, i] = 1

        self.lever_pos = [self.map_size - 2, self.map_size - 2]

        # initialize agent 1
        self.agt1_pos = [self.half_pos+1, 1]
        self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

        # initialize agent 2
        self.agt2_pos = [self.map_size-2, 1]
        self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1

        # initialize treasure
        self.treasure_pos = [1, self.map_size - 2]
        # self.treasure_pos = [self.half_pos - 1, self.half_pos]

        # sub pos = [self.map_size - 2, self.map_size - 2]
        self.sub_pos = [self.map_size - 3, self.map_size - 2]

    def reset(self):
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0, i] = 1
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.map_size - 1] = 1
            self.occupancy[self.map_size - 1, i] = 1
            self.occupancy[self.half_pos, i] = 1

        self.lever_pos = [self.map_size - 2, self.map_size - 2]

        # initialize agent positions
        self.agt1_pos = [self.half_pos + 1, 1]
        self.agt2_pos = [self.map_size - 2, 1]

        # initialize treasure position
        self.treasure_pos = [1, self.map_size - 2]

        # sub position
        self.sub_pos = [self.map_size - 3, self.map_size - 2]

        # Reset the occupancy grid with agent positions
        self.occupancy[self.agt1_pos[0], self.agt1_pos[1]] = 1
        self.occupancy[self.agt2_pos[0], self.agt2_pos[1]] = 1

        # Create the initial state for each agent
        state = np.zeros((7, 7, 3))
        state[self.agt1_pos[0], self.agt1_pos[1], 0] = 1
        state[self.agt2_pos[0], self.agt2_pos[1], 1] = 1
        state[self.treasure_pos[0], self.treasure_pos[1], 2] = 1

        return state


    def step(self, action_list):
        reward = 0
        # agent1 move
        if action_list[0] == 0:  # move up
            if self.occupancy[self.agt1_pos[0]-1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] - 1
                self.occupancy[self.agt1_pos[0]+1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 1:  # move down
            if self.occupancy[self.agt1_pos[0]+1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0]+1
                self.occupancy[self.agt1_pos[0]-1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 2:  # move left
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]-1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] - 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]+1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 3:  # move right
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]+1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] + 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]-1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1

        # agent2 move
        if action_list[1] == 0:  # move up
            if self.occupancy[self.agt2_pos[0]-1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] - 1
                self.occupancy[self.agt2_pos[0]+1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 1:  # move down
            if self.occupancy[self.agt2_pos[0]+1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] + 1
                self.occupancy[self.agt2_pos[0]-1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 2:  # move left
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]-1] != 1:  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] - 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]+1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 3:  # move right
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]+1] != 1:  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] + 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]-1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1

        # check lever
        if self.agt1_pos == self.lever_pos or self.agt2_pos == self.lever_pos:
            self.occupancy[self.half_pos][self.half_pos] = 0  # open secret door
            self.occupancy[self.half_pos][self.half_pos-1] = 0  # open secret door
            self.occupancy[self.half_pos][self.half_pos+1] = 0  # open secret door
        else:
            self.occupancy[self.half_pos][self.half_pos] = 1  # open secret door
            self.occupancy[self.half_pos][self.half_pos - 1] = 1  # open secret door
            self.occupancy[self.half_pos][self.half_pos + 1] = 1  # open secret door

        # check treasure
        if self.agt1_pos == self.treasure_pos or self.agt2_pos == self.treasure_pos:
            reward = reward + 100

        if (self.agt1_pos == self.sub_pos and self.agt2_pos == self.lever_pos) or (self.agt1_pos == self.lever_pos and self.agt2_pos == self.sub_pos):
            reward = reward + 3

        done = False
        if reward > 0:
            done = True

        return reward, done

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 0] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 1] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 2] = 0.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 0] = 0.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 1] = 1.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 2] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 0] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 1] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 2] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 0] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 1] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 2] = 1.0
        obs[self.sub_pos[0], self.sub_pos[1], 0] = 1.0
        obs[self.sub_pos[0], self.sub_pos[1], 1] = 0.0
        obs[self.sub_pos[0], self.sub_pos[1], 2] = 1.0
        return obs

    def get_agt1_obs(self):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.occupancy[self.agt1_pos[0]-1+i][self.agt1_pos[1]-1+j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                d_x = self.lever_pos[0] - self.agt1_pos[0]
                d_y = self.lever_pos[1] - self.agt1_pos[1]
                if d_x>=-1 and d_x<=1 and d_y>=-1 and d_y<=1:
                    obs[1+d_x, 1+d_y, 0] = 1.0
                    obs[1+d_x, 1+d_y, 1] = 1.0
                    obs[1+d_x, 1+d_y, 2] = 0.0
                d_x = self.treasure_pos[0] - self.agt1_pos[0]
                d_y = self.treasure_pos[1] - self.agt1_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1+d_x, 1+d_y, 0] = 0.0
                    obs[1+d_x, 1+d_y, 1] = 1.0
                    obs[1+d_x, 1+d_y, 2] = 0.0
                d_x = self.agt2_pos[0] - self.agt1_pos[0]
                d_y = self.agt2_pos[1] - self.agt1_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1 + d_x, 1 + d_y, 0] = 0.0
                    obs[1 + d_x, 1 + d_y, 1] = 0.0
                    obs[1 + d_x, 1 + d_y, 2] = 1.0
        obs[1, 1, 0] = 1.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 0.0
        return obs

    def get_agt2_obs(self):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.occupancy[self.agt2_pos[0]-1+i][self.agt2_pos[1]-1+j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                d_x = self.lever_pos[0] - self.agt2_pos[0]
                d_y = self.lever_pos[1] - self.agt2_pos[1]
                if d_x>=-1 and d_x<=1 and d_y>=-1 and d_y<=1:
                    obs[1+d_x, 1+d_y, 0] = 1.0
                    obs[1+d_x, 1+d_y, 1] = 1.0
                    obs[1+d_x, 1+d_y, 2] = 0.0
                d_x = self.treasure_pos[0] - self.agt2_pos[0]
                d_y = self.treasure_pos[1] - self.agt2_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1+d_x, 1+d_y, 0] = 0.0
                    obs[1+d_x, 1+d_y, 1] = 1.0
                    obs[1+d_x, 1+d_y, 2] = 0.0
                d_x = self.agt1_pos[0] - self.agt2_pos[0]
                d_y = self.agt1_pos[1] - self.agt2_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1 + d_x, 1 + d_y, 0] = 1.0
                    obs[1 + d_x, 1 + d_y, 1] = 0.0
                    obs[1 + d_x, 1 + d_y, 2] = 0.0
        obs[1, 1, 0] = 0.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 1.0
        return obs

    def get_obs(self):
        return [self.get_agt1_obs(), self.get_agt2_obs()]

    def get_state(self):
        state = np.zeros((1, 4))
        state[0, 0] = self.agt1_pos[0] / self.map_size
        state[0, 1] = self.agt1_pos[1] / self.map_size
        state[0, 2] = self.agt2_pos[0] / self.map_size
        state[0, 3] = self.agt2_pos[1] / self.map_size
        return state

    def plot_scene(self):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_agt1_obs())
        ax3.imshow(self.get_agt2_obs())

        plt.show()

    def render(self):

        obs = self.get_global_obs()
        enlarge = 30
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):

                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 255), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(100)

#building the environment model
class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnvironmentModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim[0] + action_dim[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.state_head = nn.Linear(32, state_dim[0])

    def forward(self, state, action):
        print("State shape before flattening:", state.shape)
        state = state.squeeze(0)  # Remove the batch dimension
        print("State shape after squeezing:", state.shape)
        state = state.permute(2, 0, 1)  # Reshape to (3, 7, 7)
        print("State shape after permuting:", state.shape)
        state = state.view(1, state.size(0), state.size(1), state.size(2))  # Add the batch dimension back
        print("State shape after reshaping:", state.shape)
        action = action.view(action.size(0), -1)  # Flatten the action
        state_action = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        next_state = self.state_head(x)
        return next_state


    #defining the I2A module
class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=5, hidden_dim=256):
        super(I2A_FindTreasure, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_len = rollout_len

        # Define the environment model
        self.env_model = EnvironmentModel(state_dim, action_dim)

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(state_dim[0] * rollout_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim[0]),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_space):
        # Check if the state has the correct shape
        if state.size(1) != 7 or state.size(2) != 7 or state.size(3) != 3:
            print("irregular state shape before passing to the env model:", state.shape)
            raise ValueError("Invalid state dimension. Expected shape: (7, 7, 3)")

        # Flatten the state tensor
        state = state.view(state.size(0), -1)

        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            # Generate a random action
            action = torch.randint(action_space, (state.size(0), 1))

            # Concatenate the state and action
            state_action = torch.cat([state, action], dim=-1)

            # Pass the concatenated state-action to the environment model
            next_state = self.env_model(state_action)

            # Store the next state in the imagined states
            imagined_states.append(next_state)

        # Concatenate the imagined states along the last dimension
        imagined_states = torch.cat(imagined_states, dim=-1)

        # Return the imagined states and the original state
        return imagined_states, state
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

# Define the namedtuple to store experiences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Function to select an action using the current policy

def select_action(model, state, output_size, epsilon):
    #if epsilon > 0.95:
    if np.random.rand() < epsilon:
        # Random action
        action = torch.randint(0, output_size[0], (1,))
    else:
        # Use I2A module to produce action
        state = torch.tensor(state, dtype=torch.float32).permute(0, 3, 1, 2)  # Reshape state tensor to (1, 3, 7, 7)
        action_space = torch.tensor([output_size[0]], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = model(state, action_space)

        m = torch.distributions.Categorical(logits=action_probs)
        action = m.sample()

    return action.item()    



# Main function to train and test the I2A agent
def main(args):
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
        print("reset enviro:", state.shape)
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0

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

            memory_agent1.push(state, action_agent1, reward, next_state1, done)
            memory_agent2.push(state, action_agent2, reward, next_state2, done)

          #TRAIN AGENTS********************************************************************************************


            #if len(memory_agent1) < args.batch_size:
                #return

            if len(memory_agent1) >= args.batch_size:

                states1, actions1, rewards1, next_states1, dones = memory_agent1.sample(args.batch_size)
                states2, actions2, rewards2, next_states2, dones = memory_agent2.sample(args.batch_size)
                print(len(states1), len(states2))

                max_state_dim1 = max(states1[0].shape[1:], next_states1[0].shape[1:])
                common_shape1 = (len(states1),) + max_state_dim1

                padded_states1 = []
                padded_next_states1 = []
                for state in states1:
                    if state.shape[1:] != (7, 7, 3):
                        continue  # Skip states with incorrect shape
                    state = torch.Tensor(state)  # Convert to a PyTorch tensor

                    padding = torch.zeros(common_shape1) - 1  # Use a negative value for padding
                    state_dim = state.dim()  # Get the number of dimensions of the state tensor

                    if state_dim != 3:
                        print("Skipping state with unexpected dimensions:", state_dim)
                        continue

                    padding[:state.shape[0], :state.shape[1], :state.shape[2]] = state.permute(2, 1, 0)
                    padded_states1.append(padding)

                if len(padded_states1) == 0:
                    continue  # No valid states, skip training
                states1 = torch.stack(padded_states1)


                max_state_dim2 = max(states2[0].shape[1:], next_states2[0].shape[1:])
                common_shape2 = (len(states2),) + max_state_dim2
                padded_states2 = []
                padded_next_states2 = []
                for state in states2:
                    if state.shape[1:] != (7, 7, 3):
                        continue  # Skip states with incorrect shape
                    state = torch.Tensor(state)  # Convert to a PyTorch tensor
                    padding = torch.zeros(common_shape2) - 1  # Use a negative value for padding
                    state_dim = state.dim()  # Get the number of dimensions of the state tensor

                    if state_dim != 3:
                        print("Skipping state with unexpected dimensions:", state_dim)
                        continue

                    padding[:state.shape[0], :state.shape[1], :state.shape[2]] = state.permute(2, 1, 0)
                    padded_states2.append(padding)

                if len(padded_states2) == 0:
                    continue  # No valid states, skip training

                states2 = torch.stack(padded_states2)

                for next_state in next_states1:
                    if next_state.shape[1:] != (7, 7, 3):
                        continue  # Skip states with incorrect shape
                    padding = torch.zeros(max_state_dim) - 1  # Use a negative value for padding
                    next_state_tensor1 = torch.Tensor(next_state)
                    padding[:next_state_tensor1.shape[0], :next_state_tensor1.shape[1], :next_state_tensor1.shape[2]] = next_state_tensor1
                    padded_next_states1.append(padding)

                if len(padded_next_states1) == 0:
                    continue  # No valid next states, skip training

                next_states1 = torch.stack(padded_next_states1)

                for next_state in next_states2:
                    if next_state.shape[1:] != (7, 7, 3):
                        continue  # Skip states with incorrect shape
                    padding = torch.zeros(max_state_dim) - 1  # Use a negative value for padding
                    next_state_tensor2 = torch.Tensor(next_state)
                    padding[:next_state_tensor2.shape[0], :next_state_tensor2.shape[1], :next_state_tensor2.shape[2]] = next_state_tensor2
                    padded_next_states2.append(padding)

                if len(padded_next_states2) == 0:
                    continue  # No valid next states, skip training

                next_states2 = torch.stack(padded_next_states2)

                actions1 = torch.LongTensor(actions1)
                actions2 = torch.LongTensor(actions2)
                rewards1 = torch.Tensor(rewards1)
                rewards2 = torch.Tensor(rewards2)
                dones = torch.Tensor(dones)
                print("state shape befoe feeding to model:", states1.shape, states2.shape)


                # Compute the current Q values for both agents
                action_probs_agent1, state_values_agent1 = model_agent1(states1, actions1)
                action_values_agent1 = action_probs_agent1.gather(1, actions1)

                action_probs_agent2, state_values_agent2 = model_agent2(states2, actions2)
                action_values_agent2 = action_probs_agent2.gather(1, actions2)

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




        # Check if the episode is finished
        if done:
            print("Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
                episode, episode_reward_agent1, episode_reward_agent2, t + 1))
            break




    # Testing the trained agents
    print("Testing the trained agents...")
    test_episodes = 10
    test_rewards_agent1 = []
    test_rewards_agent2 = []

    for episode in range(test_episodes):
        state = env.reset()
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0

        for t in count():
            action_agent1 = select_action(model_agent1, state, output_size,epsilon)
            action_agent2 = select_action(model_agent2, state, output_size,epsilon)
            reward, done = env.step(action_agent1, action_agent2)
            next_state1 = env.get_global_obs()
            next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
            next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
            next_state2 = env.get_global_obs()
            next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
            next_state2 = np.expand_dims(next_state2, axis=0)
            episode_reward_agent1 += reward
            episode_reward_agent2 += reward
            state = next_state1

            if done:
                print("Test Episode: {}, Reward Agent 1: {}, Reward Agent 2: {}, Timesteps: {}".format(
                    episode, episode_reward_agent1, episode_reward_agent2, t + 1))
                test_rewards_agent1.append(episode_reward_agent1)
                test_rewards_agent2.append(episode_reward_agent2)
                break

    print("Average test reward Agent 1: {:.2f}".format(sum(test_rewards_agent1) / test_episodes))
    print("Average test reward Agent 2: {:.2f}".format(sum(test_rewards_agent2) / test_episodes))
    env.close()
if __name__ == "__main__":
    args = get_args()
    main(args)    
