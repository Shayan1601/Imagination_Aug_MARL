import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnvModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + 1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.state_head = nn.Linear(32, state_dim)

    def forward(self, state, action):
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).repeat(state.size(0), 1)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.state_head(x)
        return next_state