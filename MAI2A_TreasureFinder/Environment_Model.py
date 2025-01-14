#building the environment model
import torch
import torch.nn as nn
import torch.nn.functional as F
from config1 import config


class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(EnvironmentModel, self).__init__()
        self.state_dim = state_dim
        self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
        # actions is passed as a one-hot matrix instead of a one-hot vector
        self.action_dim = (action_dim, *state_dim[1:])  
        self.total_input_dim = (state_dim[0] + action_dim*num_agents, *state_dim[1:])
        # Load hyperparameters from the config file
        conv1_out_channels = config["conv1_out_channels"]
        conv1_filter_size = config["conv1_filter_size"]
        conv1_stride = config["conv1_stride"]
        conv2_out_channels = config["conv2_out_channels"]
        conv2_filter_size = config["conv2_filter_size"]
        conv2_stride = config["conv2_stride"]
        fc1_out_dim = config["fc1_out_dim"]

        self.conv1 = nn.Conv2d(self.total_input_dim[0], conv1_out_channels, conv1_filter_size, conv1_stride)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, conv2_filter_size, conv2_stride)
        self.fc1_dim = self.compute_fc1_dim()
        self.fc1 = nn.Linear(self.fc1_dim, fc1_out_dim)
        self.state_head = nn.Linear(fc1_out_dim, self.flattened_state_dim)

    def forward(self, state, action1, action2):
        action1_one_hot_matrix = self.one_hot(action1)
        action2_one_hot_matrix = self.one_hot(action2)
        # Cat on second dimension (1st) since the first (0th) is the batch
        state_action = torch.cat([state, action1_one_hot_matrix, action2_one_hot_matrix], dim=1)
        state_action = state_action.to(torch.float32)
        x = F.relu(self.conv1(state_action))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        next_state = self.state_head(x)
        next_state = next_state.view(next_state.size(0), *self.state_dim)
        return next_state
    
    def compute_fc1_dim(self):
        """
        Computes the input dimension of the first fully connected layer.
        """
        x = torch.zeros(1, *self.total_input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(1, -1).size(1)

    def one_hot(self, action):
        action_one_hot_matrix = torch.zeros((action.shape[0], *self.action_dim))
        for i, ac in enumerate(action):
            action_one_hot_matrix[i, ac] = 1
        return action_one_hot_matrix