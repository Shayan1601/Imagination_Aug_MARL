#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256):
        super(I2A_FindTreasure, self).__init__()
   
        self.state_dim = state_dim
        self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
  
        self.action_dim = action_dim
        self.rollout_len = rollout_len



        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.action_dim)

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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288, action_dim[0])  # Update the input size for fc1
        

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            self.flatten,
            self.fc1,
            
        )


        #Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state, action_space):


        flattened_state = state
        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action

            action =self.distilledpolicy(flattened_state)
            

            # Pass the concatenated state-action to the environment model
            next_state = self.env_model(state, action)

            # Store the next state in the imagined states

            flattened_state= next_state
            imagined_states.append(flattened_state)
            state = next_state

        # Concatenate the imagined states along the last dimension 
        imagined_states = torch.cat(imagined_states, dim=0)
       
        
        #encoded_imagined_states = self.imagination(imagined_states)
        encoded_imagined_states = imagined_states
        
        action_prob = self.policy_head(encoded_imagined_states)

        #   Return the action prob 
        return action_prob