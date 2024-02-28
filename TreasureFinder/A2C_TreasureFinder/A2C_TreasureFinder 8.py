
#this multi agent method serve as the baseline to compare with the proposed I2A multi agent 
# the multi agent environment of choice is a 2D small grid world called Treasure Finder

#i tried to train a baseline but so far failed. either the agent model is not complicated enough
#or the training loop with loss fcn and optimizer is not doing well.add()#batch size only 5 works\


#importing the dependencies
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from stable_baselines3.common.env_util import make_vec_env
from env_FindTreasure import EnvFindTreasure
import os
import torch.nn as nn
import time
from itertools import count
import numpy as np
import torch.autograd as autograd

os.chdir('/Users/shayan/Desktop/Reboot treasure')

class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        # self.fc1_actor = nn.Linear(32*3*3, 64)
        self.fc1_actor = nn.Linear(self.feature_size(), 256)
        self.fc2_actor = nn.Linear(256, output_size[0])
        
        self.fc1_critic = nn.Linear(self.feature_size(), 256)
        self.fc2_critic = nn.Linear(256, 1)

    def forward(self, x):
        # x = torch.relu(self.conv1(x)) 
        # x = torch.relu(self.conv2(x))
        x = self.features(x)

        x = self.flatten(x)
        logits = torch.relu(self.fc1_actor(x))
        action_probs = torch.softmax(self.fc2_actor(logits), dim=1)
        # action_probs = torch.softmax(self.fc2_actor(logits), dim=-1)
        value = self.fc2_critic(torch.relu(self.fc1_critic(x)))
        return action_probs, value
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_size))).view(1, -1).size(1)



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

class A2CAgent:
    def __init__(self, input_size, output_size, learning_rate, buffer_size, batch_size, gamma):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.actor_critic = ActorCritic(input_size, output_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.actor_critic(state)
            action = torch.multinomial(action_probs, 1).item()
            action = np.array([int(action)], dtype=np.int64)
        return action
    # def update_epsilon(self, episode):
    #         self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay * episode)
            
    def update(self, states, actions, rewards, dones, next_states):
        if len(self.replay_buffer.buffer) < self.batch_size:
                return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        next_states = torch.FloatTensor(next_states)

        _, next_state_values = self.actor_critic(next_states)
        _, state_values = self.actor_critic(states)

        target_values = rewards + (1 - dones) * self.gamma * next_state_values
        advantage = target_values - state_values

        log_probs, values = self.actor_critic(states)
        actor_loss = -(log_probs.gather(1, actions) * advantage.detach()).mean()
        critic_loss = nn.functional.mse_loss(values, target_values.detach())
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # Optionally apply gradient clipping
        #nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)  # Adjust max_norm as needed
        self.optimizer.step()

if __name__ == '__main__':

    env = EnvFindTreasure(8)  # Adjust the arguments based on your environment

    input_size = (3,3,3)  # Update with the appropriate state size attribute
    output_size = (4,) #env.action_size  # Update with the appropriate action size attribute
    learning_rate = 0.0001
    gamma = 0.99
    max_time_steps = 750   
    buffer_size = 1
    batch_size = 1
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.001


    # Create A2C agents
    agents = []
    for _ in range(2):  
        agent = A2CAgent(input_size, output_size,learning_rate, buffer_size,batch_size, gamma)
        agents.append(agent)

    # Training loop
    num_episodes = 10000  # Define the number of episodes for training
    mean_rewards = []
    mean_R_Plot = []
    
    # Start time
    start_time = time.time()
    for episode in range(num_episodes):
        env.reset()
        done = False
        total_reward = 0
        state1 = env.get_agt1_obs()
        state2 = env.get_agt2_obs()
        

        for t in count():
            
            
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

                

            # Update agents
            agents[0].update(state1, action1, reward, done, next_state1)
            agents[1].update(state2, action2, reward, done, next_state2)

            state1 = next_state1
            state2 = next_state2
            total_reward += reward
            
            if done or t >= max_time_steps:
                break
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        # print(f"Episode {episode+1}:  Reward = {total_reward}")
        mean_rewards.append(total_reward)
        mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards)
        mean_R_Plot. append(mean_reward_agent1)
        if (episode + 1) % 100 == 0:
             
            print(f"Episode {episode+1}: Mean Reward = {mean_reward_agent1}")

        #print(f"Episode {episode+1}: Total Reward = {total_reward}")

    
        # Save the agents after training
    for i, agent in enumerate(agents):
        torch.save(agent.actor_critic.state_dict(), f'A2Cagent_{i}.pth')
        
        # Calculate total mean rewards
    total_mean_reward_agent1 = sum(mean_rewards) / len(mean_rewards) 
    # Print total mean rewards
    print(f"Total Mean Reward after {num_episodes} episodes : {total_mean_reward_agent1}") 
    
    # End time
    end_time = time.time()
    # Total training time
    training_time_seconds = end_time - start_time
    training_hours = int(training_time_seconds // 3600)
    training_minutes = int((training_time_seconds % 3600) // 60)
    
    print(f"Total training time: {training_hours} hours and {training_minutes} minutes")
    training_time_str = f"Training Time: {training_hours} hours {training_minutes} minutes"
    
    # Plotting
    episodes = list(range(1, num_episodes + 1, 1))
    plt.plot(episodes, mean_R_Plot, label='Mean Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Episodes')
    plt.grid(True)
    plt.legend()
    plt.text(5, min(mean_R_Plot) - 0.2, training_time_str, ha='left')
    plt.text(5, min(mean_R_Plot) + 3.8, f"Total Mean Reward: {total_mean_reward_agent1}", ha='left')
    plt.show()
          

