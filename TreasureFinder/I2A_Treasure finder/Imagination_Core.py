#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
import os
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent
from Roll_encoder import Encoder
 

class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=3, agent_mode=1):
        super(I2A_FindTreasure, self).__init__()

        self.state_dim = state_dim
        self.flattened_state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.action_dim = action_dim
        self.rollout_len = rollout_len
        self.agent_mode = agent_mode

        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim, self.state_dim, self.action_dim, self.action_dim)

        # Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))

        # Define the rollout encoder
        self.encoder = Encoder(self.state_dim, hidden_size = hidden_dim)


        # The model-free part of I2A
        # self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)


        
        # Define the policy head
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels=32, kernel_size=3, stride=1, padding=1) 



        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288, 32)
  
        self.fc2 = nn.Linear(32, action_dim[0])

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            self.conv2,
    
            
            self.flatten,  
            self.fc1,
            self.fc2,
        
        )
        # Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)
        # Load distilled policy weights
        if self.agent_mode ==1: 
            self.distilledpolicy.load_state_dict(torch.load('agent_1_distilledp.pth'))   
            
        elif self.agent_mode == 2:
            self.distilledpolicy.load_state_dict(torch.load('agent_2_distilledp.pth'))
        

    def forward(self, state1, state2, distilledpolicyp):

        # if self.agent_mode ==1:    
        #     imagined_states = [state1]
        # elif self.agent_mode == 2:
        #     imagined_states = [state2]
        state11= state1
        state22= state2   
            
        if self.agent_mode ==1:    
            imagined_states = [state1]
        elif self.agent_mode == 2:
            imagined_states = [state2]
 
        for _ in range(self.rollout_len):
            
            if self.agent_mode ==1:    
                action1 = self.distilledpolicy(state1)
                action2 = distilledpolicyp(state2)
            elif self.agent_mode == 2:
                action1 = distilledpolicyp(state1)
                action2 = self.distilledpolicy(state2)

            next_state1, next_state2 = self.env_model(state1, state2, action1, action2)
            if self.agent_mode ==1:
                imagined_states.append(next_state1)
            elif self.agent_mode == 2:
                imagined_states.append(next_state2)
            state1 = next_state1
            state2 = next_state2
        
        # passing the imagined states through the Roll-out Encoder    
        enc_out = torch.zeros((state1.size()[0], 3, 3, 3))
        for i in range(len(imagined_states) - 1, -1, -1):
            state = imagined_states[i]
            # Pass the  state and the previous encoder output through the encoder function
            enc_out = self.encoder(state,enc_out)
    

           
            

        
        # aggregating model-free and imagined states
        encoded_imagined_states = enc_out 
        if self.agent_mode ==1:    
            final_state= torch.cat((state11, encoded_imagined_states), dim=1)
            #final_state = state11
        elif self.agent_mode == 2:
            final_state= torch.cat((state22, encoded_imagined_states), dim=1)
            #final_state = state22
        
        action_prob = self.policy_head(encoded_imagined_states) 

        return action_prob