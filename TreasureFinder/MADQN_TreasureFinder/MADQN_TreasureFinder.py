
#this multi agent method serve as the baseline to compare with the proposed I2A multi agent 
# the multi agent environment of choice is a 2D small grid world called Treasure Finder

#i tried to train a baseline but so far failed. either the agent model is not complicated enough
#or the training loop with loss fcn and optimizer is not doing well.add()#batch size only 5 works\


#importing the dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from stable_baselines3.common.env_util import make_vec_env
from env_FindTreasure import EnvFindTreasure


import torch.nn as nn

import torch.nn as nn

import torch.nn as nn

class MADQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MADQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Assuming input_size is (3, 3, 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()  # Add Flatten layer
        self.fc1 = nn.Linear(128*3 * 3, 196)  # Update input size for fc1
        self.fc2 = nn.Linear(196, output_size[0])

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))  
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
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

        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_size)
        else:
            with torch.no_grad():
                state = torch.Tensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                action = np.array([int(torch.argmax(q_values, dim=1).item())], dtype=np.int64)
            #m = torch.distributions.Categorical(logits=q_values)
            #action = m.sample()
        return action

    def train(self):
            if len(self.replay_buffer.buffer) < self.batch_size:
                return

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            

            states = torch.Tensor(states)
            next_states = torch.Tensor(next_states)

            actions = torch.LongTensor(actions)
            rewards = torch.Tensor(rewards)
            dones = torch.Tensor(dones)
            
            
            




            q_values = self.q_network(states).gather(1, actions)
           
            

            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
            target_q_values = target_q_values.unsqueeze(1)


 
            
            # Using Huber loss instead of MSE loss
            loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    # Create an instance of the multi-agent environment
    env = EnvFindTreasure(10)  # Adjust the arguments based on your environment
    # def create_custom_env():
    #     # Assume bars is defined or passed as a global variable
    #     env1 = TreasureFinderEnv(map_size=7)
    #     return env1

    # env = make_vec_env(create_custom_env, seed=1, n_envs=10)
    # Set hyperparameters



    input_size = (3,3,3)  # Update with the appropriate state size attribute
    output_size = (4,) #env.action_size  # Update with the appropriate action size attribute
    buffer_size = 100000
    batch_size = 100
    discount_factor = 0.999
    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.001
    update_target_frequency = 5


    # Create MADQN agents
    agents = []
    for _ in range(2):  # Update with the appropriate number of agents attribute
        agent = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)
        agents.append(agent)

    # Training loop
    num_episodes = 10000  # Define the number of episodes for training
    mean_rewards = []
    mean_R_Plot = []
    for episode in range(num_episodes):
        env.reset()
        done = False
        total_reward = 0
        state1 = env.get_agt1_obs()
        state2 = env.get_agt2_obs()

        while not done:
            
            
            action1 = agents[0].get_action(state1, epsilon)
            action2 = agents[1].get_action(state2, epsilon)
            action_list = [action1, action2]    

            reward, done = env.step(action_list)
            
            next_state1 = env.get_agt1_obs()
            next_state2 = env.get_agt2_obs()
            
            next_states= [next_state1, next_state2 ]
            

            # print("after step in env:", next_state.shape)

            agents[0].replay_buffer.add(state1, action1, reward, next_state1, done)
            agents[1].replay_buffer.add(state2, action2, reward, next_state2, done)
                

            for agent in agents:
                agent.train()

            state1 = next_state1
            state2 = next_state2
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        if episode % update_target_frequency == 0:
            for agent in agents:
                agent.update_target_network()
        
                
        #print(f"Episode {episode+1}:  Reward = {total_reward}")
        mean_rewards.append(total_reward)
        mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards)
        mean_R_Plot. append(mean_reward_agent1)
        if (episode + 1) % 100 == 0:
             
            print(f"Episode {episode+1}: Mean Reward = {mean_reward_agent1}")

        #print(f"Episode {episode+1}: Total Reward = {total_reward}")
        
        # Save the agents after training
    for i, agent in enumerate(agents):
        torch.save(agent.q_network.state_dict(), f'agent_{i}_model.pth')
        
        # Calculate total mean rewards
    total_mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards) 
    # Print total mean rewards
    print(f"Total Mean Reward after {num_episodes} episodes : {total_mean_reward_agent1}") 

    # Plotting
    episodes = list(range(1, num_episodes + 1, 1))
    plt.plot(episodes, mean_R_Plot, label='Mean Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Episodes')
    plt.legend()
    plt.show()
          

