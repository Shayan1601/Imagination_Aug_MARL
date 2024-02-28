#importing the dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from itertools import count
from collections import namedtuple
from random import randint
import torch.nn.functional as F
from config1 import config

from DistillPolicy import DistillPolicyAgent
from config1 import hyperparameters_agent1, hyperparameters_agent2
from env_FindTreasure import EnvFindTreasure
os.chdir('/Users/shayan/Desktop/Reboot treasure/March')



class EnvironmentModel(nn.Module):
    def __init__(self, state_dim1,state_dim2, action_dim1, action_dim2 ):
        super(EnvironmentModel, self).__init__()

        # Assuming state_dim is (3, 3, 3) and action_dim is (4,)
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_state1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2_state2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, state1, state2, action1, action2):
        # Concatenate states and actions along the channel dimension
        action1 = action1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, state1.shape[2], state1.shape[3])
        action2 = action2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, state2.shape[2], state2.shape[3])

        x = torch.cat([state1, state2, action1, action2], dim=1)

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))

        # Predicted next states
        predicted_next_state1 = torch.tanh(self.conv2_state1(x))  # Assuming the output should be between -1 and 1
        predicted_next_state2 = torch.tanh(self.conv2_state2(x))  # Assuming the output should be between -1 and 1

        return predicted_next_state1, predicted_next_state2
    
# Create the environment
state_dim1 = (3, 3, 3)
state_dim2 = (3, 3, 3)
action_dim1 = (4,)
action_dim2 = (4,)



env = EnvFindTreasure(8)

#create the environment model
env_model = EnvironmentModel(state_dim1, state_dim2, action_dim1, action_dim2)



#define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(env_model.parameters(), lr=0.001)


# Defining the replay memory for both agents
Experience = namedtuple('Experience', ('state1', 'state2', 'action1_one_hot', 'action2_one_hot', 'next_state1', 'next_state2'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state1, state2, action1_one_hot, action2_one_hot, next_state1, next_state2):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state1=state1, state2=state2, 
                                                action1_one_hot=action1_one_hot, action2_one_hot=action2_one_hot,
                                                next_state1=next_state1, next_state2=next_state2)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states1, states2, actions1, actions2, next_states1, next_states2 = zip(*[self.memory[i] for i in batch])
        return list(states1), list(states2), list(actions1), list(actions2), list(next_states1), list(next_states2)

    def __len__(self):
        return len(self.memory)
    
# Creating replay memory for both agents
memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)    



# Main training loop

input_size = (3,3,3)  # Update with the appropriate state size attribute
output_size = (4,)
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.001


for episode in range(hyperparameters_agent1.num_episodes):
    state = env.reset()
    state1 = env.get_agt1_obs()
    state2 = env.get_agt2_obs()


    dones = False

    max_time_steps = 100  # Maximum number of time steps

    for t in count():
        # Select actions based on the current policies for both agents

        action1 = (randint(0,3),) 
        action2 = (randint(0,3),)
        def one_hot(action_index, num_actions=4):
            action_one_hot = np.zeros(num_actions)
            action_one_hot[action_index] = 1
            return action_one_hot
        action1_one_hot = one_hot(action1[0])
        action2_one_hot = one_hot(action2[0])
        #action3 = str(action1 * 10 + action2).zfill(2)  #concat into int
        action_list= [action1, action2]

        # Execute the actions and store the experiences for both agents
        
        reward, done = env.step(action_list)
        next_state1 = env.get_agt1_obs()
        next_state2 = env.get_agt2_obs()

        
        

        
        

        memory_agent1.push(state1,state2, action1_one_hot,action2_one_hot, next_state1,next_state2)
        

        ###############
        # TRAIN ENV Model #
        ###############

        if len(memory_agent1) >= hyperparameters_agent1.batch_size:

            states1, states2, actions1_one_hot, actions2_one_hot, next_states1, next_states2 = memory_agent1.sample(hyperparameters_agent1.batch_size)


            states1 = torch.Tensor(states1)
            states2 = torch.Tensor(states2)
            next_states1= torch.Tensor(next_states1)
            next_states2= torch.Tensor(next_states2)
            actions1_one_hot = torch.Tensor(actions1_one_hot)
            actions2_one_hot = torch.Tensor(actions2_one_hot)
            #dones = torch.Tensor(dones)


            
            # Forward pass
            
            predicted_state1, predicted_state2 = env_model(states1, states2, actions1_one_hot, actions2_one_hot )

            
            # Compute the loss
            loss1 = criterion(predicted_state1, next_states1)
            loss2 = criterion(predicted_state2, next_states2)
            total_loss = loss1 + loss2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Update the states and episode rewards for both agents
        state1 = next_state1
        state2 = next_state2
        
        


      
        if done or t >= max_time_steps:
            break
    # Print loss for monitoring
    print(f"Episode {episode + 1}/{hyperparameters_agent1.num_episodes}, Loss: {total_loss*1000000}")
        

        
# Save the trained environment model
torch.save(env_model.state_dict(), "env_model.pth")  