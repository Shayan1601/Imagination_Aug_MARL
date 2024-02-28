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
from Policy_head import ActorCritic
def one_hot_encode(input_tensor, num_classes=4):


    batch_size = input_tensor.size(0)
    
    # Create an empty tensor to store the one-hot encoded values
    one_hot_tensor = torch.zeros(batch_size, num_classes)
    
    # Iterate through each element of the input tensor
    for i in range(batch_size):
        # Extract the value from the input tensor
        value = input_tensor[i].item()
        
        # Ensure the value is within the range of num_classes
        value = min(max(0, value), num_classes - 1)
        
        # Set the corresponding index in the one-hot tensor to 1
        one_hot_tensor[i, value] = 1
    return one_hot_tensor

def one_hot(tensor):
    # Find the index of the largest element
    _, max_index = tensor.max(dim=1)
    
    # Create a one-hot encoded tensor
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor[0, max_index] = 1
    
    return one_hot_tensor

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
        # self.encoder = Encoder(self.state_dim, hidden_size = hidden_dim)


        # The model-free part of I2A
        # self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)


        
        # Define the policy head
        self.policy_head = ActorCritic(self.state_dim, self.action_dim)

        # Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)
        # Load distilled policy weights
        if self.agent_mode ==1: 
            self.distilledpolicy.load_state_dict(torch.load('agent_1_distilledp.pth'))   
            
        elif self.agent_mode == 2:
            self.distilledpolicy.load_state_dict(torch.load('agent_2_distilledp.pth'))
        

    def forward(self, state1, state2, actionp):

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
                action2 = one_hot_encode(actionp)
            elif self.agent_mode == 2:
                action1 = one_hot_encode(actionp)
                action2 = self.distilledpolicy(state2)

            next_state1, next_state2 = self.env_model(state1, state2, action1, action2)
            if self.agent_mode ==1:
                # imagined_states.append(next_state1)
                imagined_states = next_state1
            elif self.agent_mode == 2:
                # imagined_states.append(next_state2)
                imagined_states = next_state2
            state1 = next_state1
            state2 = next_state2
        
        # passing the imagined states through the Roll-out Encoder    
        # enc_out = torch.zeros((state1.size()[0], 3, 3, 3))
        # for i in range(len(imagined_states) - 1, -1, -1):
        #     state = imagined_states[i]
            
        #     # Pass the  state and the previous encoder output through the encoder function
        #     enc_out = self.encoder(state,enc_out)
    

        
        # aggregating model-free and imagined states
        # encoded_imagined_states = enc_out 
        encoded_imagined_states = imagined_states 
        if self.agent_mode ==1:    
            #final_state= torch.cat((state11, encoded_imagined_states), dim=1)
            final_state = state11
        elif self.agent_mode == 2:
            #final_state= torch.cat((state22, encoded_imagined_states), dim=1)
            final_state = state22
        
        action_prob, value = self.policy_head(final_state) 

        return action_prob, value