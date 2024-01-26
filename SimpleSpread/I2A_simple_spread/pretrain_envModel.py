#importing the dependencies

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from collections import namedtuple
from random import randint
import torch.nn.functional as F



from config1 import hyperparameters_agent1, hyperparameters_agent2

from pettingzoo.mpe import simple_spread_v3




class EnvironmentModel(nn.Module):
    def __init__(self, state_dim1,state_dim2, state_dim3, action_dim1, action_dim2, action_dim3 ):
        super(EnvironmentModel, self).__init__()

        # Assuming state_dim is (3, 3, 3) and action_dim is (4,)
        self.fc1 = nn.Linear(69, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4_state1 = nn.Linear(32, 18)
        self.fc4_state2 = nn.Linear(32, 18)
        self.fc4_state3 = nn.Linear(32, 18)
      

    def forward(self, state1, state2, state3, action1, action2, action3):
        # Concatenate states and actions along the channel dimension

        x = torch.cat([state1, state2, state3, action1, action2, action3], dim=1)

        # Apply convolutional layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Predicted next states
        predicted_next_state1 = torch.tanh(self.fc4_state1(x))  # Assuming the output should be between -1 and 1
        predicted_next_state2 = torch.tanh(self.fc4_state2(x))
        predicted_next_state3 = torch.tanh(self.fc4_state3(x))# Assuming the output should be between -1 and 1

        return predicted_next_state1, predicted_next_state2, predicted_next_state3
# Create the environment
state_dim1 = (18,)
state_dim2 = (18,)
state_dim3 = (18,)
action_dim1 = (5,)
action_dim2 = (5,)
action_dim3 = (5,)



env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=500, continuous_actions=False)

#create the environment model
env_model = EnvironmentModel(state_dim1, state_dim2, state_dim3, action_dim1, action_dim2, action_dim3)



#define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(env_model.parameters(), lr=0.0001)


# Defining the replay memory for both agents
Experience = namedtuple('Experience', ('state1', 'state2', 'state3', 'action1', 'action2','action3', 'next_state1', 'next_state2','next_state3'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state1,state2, state3, action1, action2, action3, next_state1,next_state2, next_state3):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state1=state1, state2=state2, state3=state3, action1=action1, action2=action2, action3 = action3, next_state1=next_state1, next_state2=next_state2, next_state3= next_state3)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        # TODO: take chunks of trajectories instead of single experiences
        states1, states2, states3,  actions1, actions2, actions3, next_states1, next_states2, next_states3 = zip(*[self.memory[i] for i in batch])
        return list(states1), list(states2), list(states3), list(actions1), list(actions2), list(actions3), list(next_states1), list(next_states2), list(next_states3)

    def __len__(self):
        return len(self.memory)
    
# Creating replay memory for all agents
memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)    

def one_hot(index):

    # Ensure the input index is within the valid range [0, 4]
    assert 0 <= index[0] <= 4, "Input index should be between 0 and 4"

    # Create a tensor of zeros with size (5, 1)
    #zeros_list = [[0] for _ in range(5)]

    one_hot_tensor = np.zeros(5)

    # Set the value at the specified index to 1
    one_hot_tensor[index[0]] = 1

    return one_hot_tensor


# Main training loop

input_size = (3,3,3)  # Update with the appropriate state size attribute
output_size = (4,)
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.001
mean_loss= []
total_loss=0
for episode in range(hyperparameters_agent1.num_episodes):
    observations, infos = env.reset()
    state1= observations['agent_0']
    state2= observations['agent_1']
    state3= observations['agent_2']


    #dones = False

    max_time_steps = 500  # Maximum number of time steps

    for t in count():
        # Select actions based on the current policies for both agents

        action1 = (randint(0,4),)
        actionn1 = one_hot(action1)
        action2 = (randint(0,4),)
        actionn2 = one_hot(action2)
        action3 = (randint(0,4),)
        actionn3 = one_hot(action3)
        


        action_list = {'agent_0':int(action1[0]), 'agent_1': int(action2[0]), 'agent_2':  int(action3[0])}

        # Execute the actions and store the experiences for both agents
        
        observations, reward, terminations, truncations, infos = env.step(action_list)
        next_state1= observations['agent_0']
        next_state2= observations['agent_1']
        next_state3= observations['agent_2']

        
        

        
        

        memory_agent1.push(state1,state2, state3, actionn1, actionn2, actionn3, next_state1,next_state2, next_state3)
        #memory_agent2.push(state, action_agent2, reward, next_state2, done)

        ###############
        # TRAIN ENV Model #
        ###############

        if len(memory_agent1) >= hyperparameters_agent1.batch_size:

            states1, states2, states3, actions1, actions2, actions3, next_states1, next_states2, next_states3 = memory_agent1.sample(hyperparameters_agent1.batch_size)


            states1 = torch.Tensor(states1)
            states2 = torch.Tensor(states2)
            states3 = torch.Tensor(states3)
            next_states1= torch.Tensor(next_states1)
            next_states2= torch.Tensor(next_states2)
            next_states3= torch.Tensor(next_states3)
            actions1 = torch.Tensor(actions1)
            actions2 = torch.Tensor(actions2)
            actions3 = torch.Tensor(actions3)
            #dones = torch.Tensor(dones)


            
            # Forward pass
            
            predicted_state1, predicted_state2, predicted_state3 = env_model(states1, states2, states3, actions1, actions2, actions3 )

            
            # Compute the loss
            loss1 = criterion(predicted_state1, next_states1)
            loss2 = criterion(predicted_state2, next_states2)
            loss3 = criterion(predicted_state3, next_states3)
            total_loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Update the states and episode rewards for both agents
        state1 = next_state1
        state2 = next_state2
        state3 = next_state3
    

        
        


      
        if terminations or t >= max_time_steps:
            break
        
    mean_loss.append(total_loss)
    mean_agent_loss= sum(mean_loss) / len(mean_loss)
     # Print loss for monitoring
    if (episode + 1) % 100 == 0:    
        print(f"Episode {episode + 1}/{hyperparameters_agent1.num_episodes}, Mean Loss: {mean_agent_loss}")
        

        
# Save the trained environment model
torch.save(env_model.state_dict(), "env_model.pth")  