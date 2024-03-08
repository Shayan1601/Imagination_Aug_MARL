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

def one_hot(tensor):
    # Find the index of the largest element
    _, max_index = tensor.max(dim=1)
    
    # Create a one-hot encoded tensor
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor[0, max_index] = 1
    
    return one_hot_tensor

#Define policy-head of I2A
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()

        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(6, 1000, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.Conv2d(2000, 1000, kernel_size=1, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(1024, 512, kernel_size=1, stride=2),
            # nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc1_actor = nn.Linear(1000, 500)
        self.fc2_actor = nn.Linear(500, 100)
        self.fc3_actor = nn.Linear(100, 20)
        self.fc4_actor = nn.Linear(20, output_size[0])
        
        self.fc1_critic = nn.Linear(1000, 500)
        self.fc2_critic = nn.Linear(500, 100)
        self.fc3_critic = nn.Linear(100, 10)
        self.fc4_critic = nn.Linear(10, 1)

    def forward(self, x):
  
        critic_state= x
        x = self.features(x)

        x = self.flatten(x)
        xx = torch.relu(self.fc1_actor(x))
        xxx = torch.relu(self.fc2_actor(xx))
        logits = torch.relu(self.fc3_actor(xxx))
        action_probs = torch.softmax(self.fc4_actor(logits), dim=1)
        
        # action_probs1 = one_hot(action_probs)
        
        # critic_input = torch.cat((x, action_probs1), dim=1)
        val =self.fc3_critic(self.fc2_critic(torch.relu(self.fc1_critic(x))))
        value = self.fc4_critic(val)
        return action_probs, value
    # def feature_size(self):
    #     return self.features(autograd.Variable(torch.zeros(1, *self.input_size))).view(1, -1).size(1)