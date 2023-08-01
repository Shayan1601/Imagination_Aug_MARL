#defining the the distilled policy for in order to select the action

import torch
import torch.nn as nn


class DistillPolicyAgent(nn.Module):
    def __init__(self, in_shape, num_actions):
        super(DistillPolicyAgent, self).__init__()

        self.in_shape = in_shape
        self.num_actions = num_actions

        self.fc_layers = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(in_shape)), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x





