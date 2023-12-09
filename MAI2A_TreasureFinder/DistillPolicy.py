#defining the the distilled policy for in order to select the action

import torch
import torch.nn as nn

class DistillPolicyAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DistillPolicyAgent, self).__init__()

        # Assuming input_size is (3, 3, 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten() 
        #self.fc1 = nn.Linear(32* 3 * 3, 196)  # Adjusted input size for fc1
        self.fc1 = nn.Linear(32* 3*3, 4)  # Adjusted output size for fc2

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))  
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        
        
        return x

# class DistillPolicyAgent(nn.Module):
#     def __init__(self, in_shape, num_actions):
#         super(DistillPolicyAgent, self).__init__()

#         self.in_shape = in_shape
#         self.num_actions = num_actions

#         self.fc_layers = nn.Sequential(
#             #nn.Linear(torch.prod(torch.tensor(in_shape)), 256),
#             nn.Linear(9, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_actions)
#         )

#     def forward(self, x):
#         x = x.to(torch.float32)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         x = torch.softmax(x, dim=-1)
#         x = torch.argmax(x, dim=-1)
#         return x





