#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

class I2A_FindTreasure1(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256):
        super(I2A_FindTreasure1, self).__init__()
   
        self.state_dim = state_dim
        #self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
        self.flattened_state_dim = state_dim
  
        self.action_dim = action_dim
        self.rollout_len = rollout_len



        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim,self.state_dim, self.action_dim, self.action_dim)
        
        #Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))
        
        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # the model free part of I2A
        self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)

        # # Define the policy head

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=128, kernel_size=5, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        #self.gru = nn.GRU(input_size=32 * 3 * 3, hidden_size=288, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 256)  # Update the input size for fc1
        self.fc2 = nn.Linear(256, action_dim[0])
        

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            #self.conv2,
            self.flatten,
            self.fc1,
            self.fc2,
        )


        #Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state1, state2, action_space2):


        
        def one_hot(action_indices, num_actions=4):
            # Ensure action_indices are within bounds
            action_indices = action_indices.long() % num_actions
            
            # Create one-hot vectors for each element in the batch
            batch_size = action_indices.size(0)
            action_one_hots = torch.zeros(batch_size, num_actions)
            action_indices = action_indices.view(batch_size, 1)  # Ensure indices have shape (batch_size, 1)
            action_one_hots.scatter_(1, action_indices, 1)
    
            return action_one_hots
        
        action_space2 = one_hot(action_space2)
        flattened_state = state1
        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action

            action1 =self.distilledpolicy(flattened_state)
            

            # Pass the concatenated state-action to the environment model
            
            next_state, _ = self.env_model(state1, state2, action1, action_space2)

            # Store the next state in the imagined states

            flattened_state= next_state
            imagined_states.append(flattened_state)
            #state1 = next_state

        # Concatenate the imagined states along the last dimension 
        imagined_states = torch.cat((state1, next_state), dim=1)
       
        
        #encoded_imagined_states = self.imagination(imagined_states)
        encoded_imagined_states = imagined_states
        
        action_prob = self.policy_head(encoded_imagined_states)

        #   Return the action prob 
        return action_prob
    

class I2A_FindTreasure2(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256):
        super(I2A_FindTreasure2, self).__init__()
   
        self.state_dim = state_dim
        #self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
        self.flattened_state_dim = state_dim
  
        self.action_dim = action_dim
        self.rollout_len = rollout_len



        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim,self.state_dim, self.action_dim, self.action_dim)
        
        #Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))
        
        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # the model free part of I2A
        self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)

        # # Define the policy head

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=128, kernel_size=5, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        #self.gru = nn.GRU(input_size=32 * 3 * 3, hidden_size=288, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 256)  # Update the input size for fc1
        self.fc2 = nn.Linear(256, action_dim[0])

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            #self.conv2,
            self.flatten,
            self.fc1,
            self.fc2,
            
        )


        #Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state1, state2, action_space1):
        

                
        def one_hot(action_indices, num_actions=4):
            # Ensure action_indices are within bounds
            action_indices = action_indices.long() % num_actions
            
            # Create one-hot vectors for each element in the batch
            batch_size = action_indices.size(0)
            action_one_hots = torch.zeros(batch_size, num_actions)
            action_indices = action_indices.view(batch_size, 1)  # Ensure indices have shape (batch_size, 1)
            action_one_hots.scatter_(1, action_indices, 1)
    
            return action_one_hots
        action_space1 = one_hot(action_space1)
        flattened_state = state2
        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action

            action2 =self.distilledpolicy(flattened_state)
            #action2 = action2.squeeze(0)

            # Pass the concatenated state-action to the environment model
            
            _ , next_state = self.env_model(state1, state2, action2, action_space1)

            # Store the next state in the imagined states

            flattened_state= next_state
            imagined_states.append(flattened_state)
            #state2 = next_state

        # Concatenate the imagined states along the last dimension 
        imagined_states = torch.cat((state2, next_state), dim=1)
        

       
        
        #encoded_imagined_states = self.imagination(imagined_states)
        encoded_imagined_states = imagined_states
        
        action_prob = self.policy_head(encoded_imagined_states)

        #   Return the action prob 
        return action_prob