#model free part of the algorithm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from IPython.display import clear_output

import matplotlib.pyplot as plt
#%matplotlib inline

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

#Nueral Network model for A2C
# class ModelFreeAgent(nn.Module):
#     def __init__(self, in_shape, num_actions, hidden_dim=256):
#         super(ModelFreeAgent, self).__init__()
        
#         self.in_shape = in_shape

#         self.fc_layers = nn.Sequential(
#             nn.Linear(self.in_shape, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#         )
     
#         self.fc = nn.Sequential(
#             nn.Linear(16, hidden_dim),
#             nn.ReLU(),
#         )
        
#         self.critic = nn.Linear(hidden_dim, 1)
#         self.actor = nn.Linear(hidden_dim, num_actions)

      


#     def forward(self, x):
#         x = torch.tensor(x)
#         x = x.to(torch.float32)
#         x = self.fc_layers(x)
#         x = self.fc(x)
#         return x
import torch.nn as nn

class ModelFreeAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelFreeAgent, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Assuming input_size is (3, 3, 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()  # Add Flatten layer
        self.fc1 = nn.Linear(64*3 * 3, 196)  # Update input size for fc1
        self.fc2 = nn.Linear(196, output_size[0])

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))  
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


