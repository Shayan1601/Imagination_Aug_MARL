#building the environment model
import torch
import torch.nn as nn
import torch.nn.functional as F
class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnvironmentModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = 1

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 32)
        self.state_head = nn.Linear(32, self.state_dim)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.to(torch.float32)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        next_state = self.state_head(x)
        return next_state