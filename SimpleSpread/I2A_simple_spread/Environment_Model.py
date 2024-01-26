#building the environment model
import torch
import torch.nn as nn
import torch.nn.functional as F
from config1 import config

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
    
