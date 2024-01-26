import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config1 import hyperparameters_agent1, hyperparameters_agent2
from pettingzoo.mpe import simple_spread_v3


class MADQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MADQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Assuming input_size is (18, 1)
        self.fc1 = nn.Linear(input_size[0], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size[0])

    def forward(self, x):
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
  
            q_values = self.q_network(states).gather(1, torch.Tensor(actions).unsqueeze(1))

            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
            target_q_values = target_q_values.unsqueeze(1)
           
            # Using Huber loss instead of MSE loss
            loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


input_size = (18,)  # Update with the appropriate state size attribute
output_size = (5,) #env.action_size  # Update with the appropriate action size attribute
buffer_size = 100000
batch_size = 120
discount_factor = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.001
update_target_frequency = 5

if __name__ == '__main__':
    # Create Env and MADQN agents
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=500, continuous_actions=False)
    # Initialize step counter and maximum number of steps
    step_counter = 0
    max_steps = 500

    agents = []
    for _ in range(3):  # Update with the appropriate number of agents attribute
        agent = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)
        agents.append(agent)
        
    # Training loop
    num_episodes = 100000  # Define the number of episodes for training
    mean_rewards = []
    running_reward = 0
    mean_R_Plot = []
    for episode in range(num_episodes):
        
        total_reward1 = 0
        total_reward2 = 0
        total_reward3 = 0

        observations, infos = env.reset()
        state1= observations['agent_0']
        state2= observations['agent_1']
        state3= observations['agent_2']





        while env.agents and step_counter < max_steps:

            # this is where you would insert your policy
            
            action1 = agents[0].get_action(state1, epsilon)
            action2 = agents[1].get_action(state2, epsilon)
            action3 = agents[2].get_action(state3, epsilon)
            action_list = {'agent_0':int(action1), 'agent_1': int(action2), 'agent_2':  int(action3)} 

            observations, reward, terminations, truncations, infos = env.step(action_list)

            next_state1= observations['agent_0']
            next_state2= observations['agent_1']
            next_state3= observations['agent_2']
            
            agents[0].replay_buffer.add(state1, action_list['agent_0'], reward['agent_0'], next_state1, terminations['agent_0'])
            agents[1].replay_buffer.add(state2, action_list['agent_1'], reward['agent_1'], next_state2, terminations['agent_1'])
            agents[2].replay_buffer.add(state3, action_list['agent_2'], reward['agent_2'], next_state3, terminations['agent_2'])
            
            for agent in agents:
                agent.train()

            state1 = next_state1
            state2 = next_state2
            state3 = next_state3
            total_reward1 += reward['agent_0']
            total_reward2 += reward['agent_1']
            total_reward3 += reward['agent_2']
            # Increment the step counter
            step_counter += 1
            
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
            


        if episode % update_target_frequency == 0:
            for agent in agents:
                agent.update_target_network()
                
        mean_rewards.append(total_reward1)
        mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards)
            
        # Calculate mean reward after each 10 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Mean Reward = {mean_reward_agent1}")
    # Save the agents after training
    for i, agent in enumerate(agents):
        torch.save(agent.q_network.state_dict(), f'agent_{i}_model.pth')

    # Calculate total mean rewards
    total_mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards)     
    # Print total mean rewards
    print(f"Total Mean Reward after {num_episodes} episodes : {total_mean_reward_agent1}") 

