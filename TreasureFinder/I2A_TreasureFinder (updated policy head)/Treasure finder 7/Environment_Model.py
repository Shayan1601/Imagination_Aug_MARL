#building the environment model
import torch
import torch.nn as nn
import torch.nn.functional as F
from config1 import config

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
    
