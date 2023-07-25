#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=5, hidden_dim=256):
        super(I2A_FindTreasure, self).__init__()
        self.state_dim = np.prod(state_dim)
        self.action_dim = action_dim[0]
        self.rollout_len = rollout_len

        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.action_dim)

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(self.state_dim * rollout_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # TODO miss the model free part of I2A
#         self.model_free = nn.Sequential(
#             nn.Linear(self.state_dim, hidden_dim),
#             nn.ReLU(),
#        )

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim[0]),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_space):
        # Check if the state has the correct shape
        if state.shape[1] != 3 or state.shape[2] != 7 or state.shape[3] != 7:
            print("irregular state shape before passing to the env model:", state.shape)
            raise ValueError("Invalid state dimension. Expected shape: (3, 7, 7)")

        # Flatten the state tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = torch.reshape(state , (state.shape[0], -1))

        # compute the model free part
        #model_free_hidden = self.model_free(state)

        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            # TODO: use a distilled policy to generate actions instead of random actions
            # Generate a random action
            action = torch.randint(self.action_dim, (state.shape[0], 1))

            # Pass the concatenated state-action to the environment model
            next_state = self.env_model(state, action)

            # Store the next state in the imagined states
            imagined_states.append(next_state)
            state = next_state

        # Concatenate the imagined states along the last dimension
        imagined_states = torch.cat(imagined_states, dim=-1)
        encoded_imagined_states = self.imagination(imagined_states)
        
        # Concatenate the model free hidden state and the encoded imagined states
        #full_hidden = torch.cat([model_free_hidden, encoded_imagined_states], dim=-1)

        action_prob = self.policy_head(encoded_imagined_states)
        value = self.value_head(encoded_imagined_states)
        #   Return the action prob and the state value
        return action_prob, value