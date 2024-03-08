#defining the the distilled policy for in order to select the action

import torch
import torch.nn as nn

# class DistillPolicyAgent(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DistillPolicyAgent, self).__init__()

#         # Assuming input_size is (3, 3, 3)
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.flatten = nn.Flatten()  # Add Flatten layer
#         self.fc1 = nn.Linear(128*3 * 3, 196)  # Update input size for fc1
#         self.fc2 = nn.Linear(196, 4)

#     def forward(self, x):
        
#         x = torch.relu(self.conv1(x))  
#         x = torch.relu(self.conv2(x))
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
        
        
#         return x

class DistillPolicyAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DistillPolicyAgent, self).__init__()

        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 500, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(500, 250, kernel_size=1, stride=2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc1_actor = nn.Linear(250, 100)
        # self.fc1_actor = nn.Linear(self.feature_size(), 256)
        self.fc2_actor = nn.Linear(100, output_size[0])
        
        self.fc1_critic = nn.Linear(250, 100)
        self.fc2_critic = nn.Linear(100, 30)
        self.fc3_critic = nn.Linear(30, 1)

    def forward(self, x):
 
        x = self.features(x)

        x = self.flatten(x)
        logits = torch.relu(self.fc1_actor(x))
        action_probs = torch.softmax(self.fc2_actor(logits), dim=1)

        value =self.fc3_critic(self.fc2_critic(torch.relu(self.fc1_critic(x))))
        return action_probs, value





