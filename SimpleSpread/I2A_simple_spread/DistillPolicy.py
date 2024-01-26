#defining the the distilled policy for in order to select the action

import torch
import torch.nn as nn

class DistillPolicyAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DistillPolicyAgent, self).__init__()

        # Assuming input_size is (3, 3, 3)
        self.fc1 = nn.Linear(state_dim[0], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_dim[0])
        


    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(x, dim=-1)
       
        
        return x







