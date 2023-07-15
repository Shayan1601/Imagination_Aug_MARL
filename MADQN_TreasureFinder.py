
#this multi agent method serve as the baseline to compare with the proposed I2A multi agent 
# the multi agent environment of choice is a 2D small grid world called Treasure Finder


#importing the dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

#Defining the model network of the agents
class MADQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MADQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        flattened_size = input_size[-1] * input_size[-2] * input_size[-3]
        self.fc1 = nn.Linear(flattened_size, 196)
        self.fc2 = nn.Linear(196, 196)
        self.fc3 = nn.Linear(196, output_size[0])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

#defining the multi agent class

class MADQNAgent:
    def __init__(self, input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_network = MADQN(input_size, output_size)
        self.target_network = MADQN(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state, epsilon):
        if state.shape != (1, 196):
            return np.random.randint(self.output_size)

        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_size)
        else:
            with torch.no_grad():
                state = torch.Tensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def train(self):
            if len(self.replay_buffer.buffer) < self.batch_size:
                return

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            max_state_dim = max(states[0].shape[1:], next_states[0].shape[1:])
            common_shape = (len(states),) + max_state_dim

            padded_states = []
            padded_next_states = []
            for state in states:
                if state.shape[1:] != (7, 7, 3):
                    continue  # Skip states with incorrect shape
                state = torch.Tensor(state)  # Convert to a PyTorch tensor
                padding = torch.zeros(max_state_dim) - 1  # Use a negative value for padding
                padding[:state.shape[0], :state.shape[1], :state.shape[2]] = state
                padded_states.append(padding)

            if len(padded_states) == 0:
                return  # No valid states, skip training

            states = torch.stack(padded_states)

            for next_state in next_states:
                if next_state.shape[1:] != (7, 7, 3):
                    continue  # Skip states with incorrect shape
                padding = torch.zeros(max_state_dim) - 1  # Use a negative value for padding
                next_state_tensor = torch.Tensor(next_state)
                padding[:next_state_tensor.shape[0], :next_state_tensor.shape[1], :next_state_tensor.shape[2]] = next_state_tensor
                padded_next_states.append(padding)

            if len(padded_next_states) == 0:
                return  # No valid next states, skip training

            next_states = torch.stack(padded_next_states)

            actions = torch.LongTensor(actions)
            rewards = torch.Tensor(rewards)
            dones = torch.Tensor(dones)

            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

#Defining the envrionemnet class
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

        # Return the initial state for each agent
        state = np.stack([
            np.copy(self.occupancy),
            np.copy(self.occupancy),
            np.copy(self.occupancy),
            np.copy(self.occupancy)
        ], axis=0)

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

        
        # Create an instance of the multi-agent environment
env = EnvFindTreasure(7)  # Adjust the arguments based on your environment

# Set hyperparameters



input_size = (7, 7, 3)  # Update with the appropriate state size attribute
output_size = (4,) #env.action_size  # Update with the appropriate action size attribute
buffer_size = 10000
batch_size = 64
discount_factor = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
update_target_frequency = 10


# Create MADQN agents
agents = []
for _ in range(2):  # Update with the appropriate number of agents attribute
    agent = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)
    agents.append(agent)

# Training loop
num_episodes = 5000  # Define the number of episodes for training
mean_rewards = []
for episode in range(num_episodes):
    state = env.reset()
    env.render()
    done = False
    total_reward = 0

    while not done:
        actions = []
        for agent in agents:
            action = agent.get_action(state, epsilon)
            actions.append(action)

        reward, done = env.step(actions)
        env.render()
        next_state = env.get_global_obs()
        next_state = np.transpose(next_state, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state = np.expand_dims(next_state, axis=0)  # Add an extra dimension for the batch

        for i, agent in enumerate(agents):
            agent.replay_buffer.add(state, actions[i], reward, next_state, done)

        for agent in agents:
            agent.train()

        state = next_state
        total_reward += reward

    epsilon = max(epsilon * epsilon_decay, min_epsilon)

    if episode % update_target_frequency == 0:
        for agent in agents:
            agent.update_target_network()

    mean_rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        mean_reward = np.mean(mean_rewards[-100:])
        print(f"Episode {episode+1}: Mean Reward = {mean_reward}")

    print(f"Episode {episode+1}: Total Reward = {total_reward}")



