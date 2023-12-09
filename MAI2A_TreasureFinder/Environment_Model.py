#building the environment model
import torch
import torch.nn as nn
import torch.nn.functional as F
from config1 import config

class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnvironmentModel, self).__init__()

        # Assuming state_dim is (1, 3, 3, 3) and action_dim is (1, 4)
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
       
        
    def forward(self, state, action):
        
        action = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, state.shape[2], state.shape[3])
        # Concatenate state and action along the channel dimension
        x = torch.cat([state, action], dim=1)

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))

        predicted_next_state = torch.tanh(self.conv2(x))  # Assuming the output should be between -1 and 1

        return predicted_next_state
    
