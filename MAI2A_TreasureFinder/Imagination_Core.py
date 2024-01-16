#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256, agent_mode=1):
        super(I2A_FindTreasure, self).__init__()

        self.state_dim = state_dim
        self.flattened_state_dim = state_dim

        self.action_dim = action_dim
        self.rollout_len = rollout_len
        self.agent_mode = agent_mode

        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.state_dim, self.action_dim, self.action_dim)

        # Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))

        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # The model-free part of I2A
        self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)

        # Define the policy head
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, action_dim[0])

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            self.flatten,
            self.fc1,
            self.fc2,
        )

        # Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state1, state2, action_space):
        def one_hot(action_indices, num_actions=4):
            action_indices = action_indices.long() % num_actions
            batch_size = action_indices.size(0)
            action_one_hots = torch.zeros(batch_size, num_actions)
            action_indices = action_indices.view(batch_size, 1)
            action_one_hots.scatter_(1, action_indices, 1)
            return action_one_hots

        action_space = one_hot(action_space)

        if self.agent_mode == 1:
            flattened_state = state1
        elif self.agent_mode == 2:
            flattened_state = state2
        else:
            raise ValueError("Invalid agent mode. Use 1 or 2.")

        imagined_states = []
        for _ in range(self.rollout_len):
            action = self.distilledpolicy(flattened_state)

            if self.agent_mode == 1:
                _, next_state = self.env_model(state1, state2, action, action_space)
            elif self.agent_mode == 2:
                _, next_state = self.env_model(state1, state2, action_space, action)
            else:
                raise ValueError("Invalid agent mode. Use 1 or 2.")

            flattened_state = next_state
            imagined_states.append(flattened_state)

        imagined_states = torch.cat((state1, next_state), dim=1)

        encoded_imagined_states = imagined_states
        action_prob = self.policy_head(encoded_imagined_states)

        return action_prob