#importing the dependencies
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from stable_baselines3.common.env_util import make_vec_env
from env_FindTreasure import EnvFindTreasure
import os
import torch.nn as nn
import time
from itertools import count
import numpy as np
import torch.autograd as autograd

class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.Conv2d(32, 16, kernel_size=1, stride=2),
            # nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        # self.fc1_actor = nn.Linear(64*3*3, 64)
        self.fc1_actor = nn.Linear(self.feature_size(), 128)
        self.fc2_actor = nn.Linear(128, output_size[0])
        
        self.fc1_critic = nn.Linear(self.feature_size(), 128)
        self.fc2_critic = nn.Linear(128, 1)

    def forward(self, x):
        # x = torch.relu(self.conv1(x)) 
        # x = torch.relu(self.conv2(x))
        x = self.features(x)

        x = self.flatten(x)
        logits = torch.relu(self.fc1_actor(x))
        action_probs = torch.softmax(self.fc2_actor(logits), dim=1)
        # action_probs = torch.softmax(self.fc2_actor(logits), dim=-1)
        value = self.fc2_critic(torch.relu(self.fc1_critic(x)))
        return action_probs, value
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_size))).view(1, -1).size(1)