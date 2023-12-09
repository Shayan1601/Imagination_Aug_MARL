#importing the dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from collections import namedtuple
from random import randint

import torch.nn.functional as F
from config1 import config


from Treasure_Finder_gymformat import TreasureFinderEnv
from DistillPolicy import DistillPolicyAgent
from Environment_Model import EnvironmentModel
from config1 import hyperparameters_agent1, hyperparameters_agent2




# Create the environment
state_dim = (7, 7, 3)
action_dim = 5

state_dim = (state_dim[-1], *state_dim[:2])
#state_dim = state_dim[0] * state_dim[1] * state_dim[2] #flattened


env = TreasureFinderEnv(7)
#create distilled policy
#distilled =DistillPolicyAgent(state_dim, action_dim)
#create the environment model
env_model = EnvironmentModel(state_dim, action_dim)



#define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(env_model.parameters(), lr=0.001)


# Defining the replay memory for both agents
Experience = namedtuple('Experience', ('state', 'action_list', 'next_state', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action_list, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state=state, action_list=action_list, next_state=next_state, done=done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        # TODO: take chunks of trajectories instead of single experiences
        states, actions, next_states, dones = zip(*[self.memory[i] for i in batch])
        return list(states), list(actions), list(next_states), list(dones)

    def __len__(self):
        return len(self.memory)
    
# Creating replay memory for both agents
memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)    

 # Main training loop


input_size = (7, 7, 3)  # Update with the appropriate state size attribute
output_size = 5
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01


for episode in range(hyperparameters_agent1.num_episodes):
    state = env.reset()
    state = np.swapaxes(state, 2, 0)
    state = np.expand_dims(state, axis=0)
    #print("reset environment:", state.shape) #(1,3,7,7)
    dones = False
    #episode_reward_agent1 = 0
    #episode_reward_agent2 = 0
    max_time_steps = 100  # Maximum number of time steps

    for t in count():
        # Select actions based on the current policies for both agents

        action1 = (randint(0,5),) 
        action2 = (randint(0,5),) 

        # Execute the actions and store the experiences for both agents
        action_list= [action1, action2]
        next_state1,reward, done = env.step(action_list)

        
        
        #print("state shape right after GET_GlobaL",next_state1.shape) --> (7,7,3)
        next_state1 = np.transpose(next_state1, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        next_state1 = np.expand_dims(next_state1, axis=0)  # Add an extra dimension for the batch
        #next_state2 = next_state1
        #next_state2 = np.transpose(next_state2, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
        #next_state2 = np.expand_dims(next_state2, axis=0)
        #print("next state shape right after GET_GlobaL",next_state1.shape) (1,3,7,7)
        
        

        if state.shape != (1,3,7,7) and next_state1 != (1,3,7,7):
            print("ERROR")
            break
        memory_agent1.push(state, action_list, next_state1, done)
        #memory_agent2.push(state, action_agent2, reward, next_state2, done)

        ###############
        # TRAIN ENV Model #
        ###############

        if len(memory_agent1) >= hyperparameters_agent1.batch_size:

            states1, actions_list, next_state1, dones = memory_agent1.sample(hyperparameters_agent1.batch_size)
            #states2, actions2, rewards2, next_states2, dones = memory_agent2.sample(hyperparameters_agent2.batch_size)
            #print("state shape right after SAMPLING",states1) #-> 60
            #print(states1.shape)
            #Trajectory = [states1, actions_list, next_states1]

            
            for i, state in enumerate(states1):
                #print(state.shape)
                states1[i] = torch.Tensor(state)  # Convert to a PyTorch tensor
            states1 = torch.cat(states1)


            next_state_tensor1 = []
            for next_state in next_state1:
                next_state_tensor1.append(torch.Tensor(next_state))

            next_states1 = torch.cat(next_state_tensor1)


            actions_list = torch.LongTensor(actions_list)
            #actions2 = torch.LongTensor(actions2)
            #rewards1 = torch.Tensor(rewards1)
            #rewards2 = torch.Tensor(rewards2)
            dones = torch.Tensor(dones)

            #print("state shape befoe feeding to model:", states1.shape, states2.shape) --> ([60,3,7,7])
            
            # Forward pass
            
            predicted_state = env_model(states1, actions_list)

            
            # Compute the loss
            loss = criterion(predicted_state, next_states1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the states and episode rewards for both agents
        state = next_state1
        
        


        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        if done or t >= max_time_steps:
            break
    # Print loss for monitoring
    print(f"Episode {episode + 1}/{hyperparameters_agent1.num_episodes}, Loss: {loss.item}")
        

        
# Save the trained environment model
torch.save(env_model.state_dict(), "env_model.pth")  



