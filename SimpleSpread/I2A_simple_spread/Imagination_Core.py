#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

class I2A_simplespread(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256, agent_mode=1):
        super(I2A_simplespread, self).__init__()

        self.state_dim = state_dim
        self.flattened_state_dim = state_dim

        self.action_dim = action_dim
        self.rollout_len = rollout_len
        self.agent_mode = agent_mode

        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.state_dim, self.state_dim, self.action_dim, self.action_dim, self.action_dim)

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
        
        # Assuming input_size is (18, 1)
        self.fc1 = nn.Linear((rollout_len+1)*state_dim[0], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim[0])

        # Define the policy head
        self.policy_head = nn.Sequential(
            
            self.fc1,
            self.fc2,
            self.fc3,
        )

        # Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)
        #self.distilledpolicy2 = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state1, state2, state3, distilledpolicyp, distilledpolicyz):


        # if self.agent_mode == 1:
        #     flattened_state = state1
        # elif self.agent_mode == 2:
        #     flattened_state = state2
        # else:
        #     raise ValueError("Invalid agent mode. Use 1 or 2.")

        
        if self.agent_mode ==1:    
            imagined_states = [state1]
        elif self.agent_mode == 2:
            imagined_states = [state2]
        elif self.agent_mode == 3:
            imagined_states = [state3]
            
        
        # # Set requires_grad to False for all parameters in the other agent's distilled policy
        # for param in distilledpolicyp.parameters():
        #     param.requires_grad = False
        # for param in self.distilledpolicy.parameters():
        #     param.requires_grad = True

        for _ in range(self.rollout_len):
            
            
            if self.agent_mode ==1:    
                action1 = self.distilledpolicy(state1)
                action2 = distilledpolicyp(state2)
                action3 = distilledpolicyz(state3)
            elif self.agent_mode == 2:
                action1 = distilledpolicyp(state1)
                action2 = self.distilledpolicy(state2)
                action3 = distilledpolicyp(state3)
            elif self.agent_mode == 3:
                action2 = distilledpolicyp(state2)
                action3 = self.distilledpolicy(state3)
                action1 = distilledpolicyp(state1)

            next_state1, next_state2, next_state3 = self.env_model(state1, state2, state3, action1, action2, action3)
            if self.agent_mode ==1:
                imagined_states.append(next_state1)
            elif self.agent_mode == 2:
                imagined_states.append(next_state2)
            elif self.agent_mode == 3:
                imagined_states.append(next_state3)
            state1 = next_state1
            state2 = next_state2
            state3 = next_state3

        imagined_obs = torch.cat( imagined_states, dim=1)

        encoded_imagined_states = imagined_obs
        action_prob = self.policy_head(encoded_imagined_states)

        return action_prob