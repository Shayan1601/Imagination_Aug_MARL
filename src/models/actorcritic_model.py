import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.env_model import EnvModel

class ActorCriticModel(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len, hidden_dim=256,activation=nn.ReLU):
        super(ActorCriticModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_len = rollout_len

        # Define the environment model
        self.env_model = EnvModel(state_dim, action_dim)

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(state_dim * rollout_len, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_space):
        # Rollout the environment model
        imagined_states = []
        for _ in range(self.rollout_len):
            action = action_space.sample()
            state = self.env_model(state, action)
            imagined_states.append(state)
        imagined_states = torch.cat(imagined_states, dim=-1)

        # Encode the imagined states
        x = self.imagination(imagined_states)

        # Compute the action probabilities
        action_probs = self.policy_head(x)

        # Compute the state values
        state_values = self.value_head(x)

        return action_probs, state_values