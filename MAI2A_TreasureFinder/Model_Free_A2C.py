#model free part of the algorithm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from IPython.display import clear_output
from Treasure_FinderEnv import EnvFindTreasure
import matplotlib.pyplot as plt
#%matplotlib inline

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

#Nueral Network model for A2C
class ModelFreeAgent(nn.Module):
    def __init__(self, in_shape, num_actions, hidden_dim=256):
        super(ModelFreeAgent, self).__init__()
        
        self.in_shape = in_shape

        self.fc_layers = nn.Sequential(
            nn.Linear(self.in_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
     
        self.fc = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
        )
        
        self.critic = nn.Linear(hidden_dim, 1)
        self.actor = nn.Linear(hidden_dim, num_actions)

      


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc_layers(x)
        x = self.fc(x)
        return x



