#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=5, hidden_dim=256):
        super(I2A_FindTreasure, self).__init__()
        self.state_dim = (state_dim[-1], *state_dim[:2])
        self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
        self.action_dim = action_dim[0]
        self.rollout_len = rollout_len

        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.action_dim)

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(self.flattened_state_dim * rollout_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # the model free part of I2A
        self.model_free_agent = ModelFreeAgent(self.flattened_state_dim, self.action_dim)

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(2*hidden_dim, action_dim[0]),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(2*hidden_dim, 1)
        
        #Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state, action_space):
        # Check if the state has the correct shape
        if state.shape[1] != 3 or state.shape[2] != 7 or state.shape[3] != 7:
            print("irregular state shape before passing to the env model:", state.shape)
            raise ValueError("Invalid state dimension. Expected shape: (3, 7, 7)")

        # Flatten the state tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        flattened_state = torch.reshape(state , (state.shape[0], -1))

        # compute the model free part
        #model_free_hidden = self.model_free(state)
        output_state_representation = self.model_free_agent(flattened_state)

        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action
            action =self.distilledpolicy(flattened_state)
            

            # Pass the concatenated state-action to the environment model
            next_state = self.env_model(state, action)

            # Store the next state in the imagined states
            flattened_state = torch.reshape(next_state, (next_state.size(0), -1))
            imagined_states.append(flattened_state)
            state = next_state

        # Concatenate the imagined states along the last dimension
        imagined_states = torch.cat(imagined_states, dim=-1)
        encoded_imagined_states = self.imagination(imagined_states)
        
        # Concatenate the model free hidden state and the encoded imagined states
        full_hidden = torch.cat([output_state_representation, encoded_imagined_states], dim=-1)

        action_prob = self.policy_head(full_hidden)
        value = self.value_head(full_hidden)
        #   Return the action prob and the state value
        return action_prob, value