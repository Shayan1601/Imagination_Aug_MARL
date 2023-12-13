#defining Imagination core
#defining the I2A module
import torch
import torch.nn as nn
import numpy as np
from Environment_Model import EnvironmentModel
from Model_Free_A2C import ModelFreeAgent
from DistillPolicy import DistillPolicyAgent

<<<<<<< Updated upstream
class I2A_FindTreasure(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, rollout_len=5, hidden_dim=256):
        super(I2A_FindTreasure, self).__init__()
        self.state_dim = (state_dim[-1], *state_dim[:2])
=======
class I2A_FindTreasure1(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256):
        super(I2A_FindTreasure1, self).__init__()
   
        self.state_dim = state_dim
>>>>>>> Stashed changes
        self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
        self.action_dim = action_dim[0]
        self.rollout_len = rollout_len

        # Define the environment model
<<<<<<< Updated upstream
        self.env_model = EnvironmentModel(self.state_dim, self.action_dim, num_agents)

=======
        self.env_model = EnvironmentModel(self.state_dim,self.state_dim, self.action_dim, self.action_dim)
        
        #Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))
        
>>>>>>> Stashed changes
        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(self.flattened_state_dim * rollout_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # the model free part of I2A
        self.model_free_agent = ModelFreeAgent(self.flattened_state_dim, self.action_dim)

        # Define the policy head
        self.policy_head = nn.Sequential(
            nn.Linear(2*hidden_dim, action_dim[0]),
            nn.Softmax(dim=-1),
        )

        # Define the value head
        self.value_head = nn.Linear(2*hidden_dim, 1)
        
        # Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

<<<<<<< Updated upstream
    def forward(self, state, action_space, other_agent_distilled_policies):
        """
        The forward pass of the I2A module.
        :param state: The state tensor of shape (batch_size, 3, 7, 7)
        :param action_space: The action space of the environment
        :param other_agent_distilled_policies: The distilled policies of the other agents
        :return: The action prob and the state value (batch_size, action_space) and (batch_size, 1)
        """
        # Check if the state has the correct shape
        if state.shape[1] != 3 or state.shape[2] != 7 or state.shape[3] != 7:
            print("irregular state shape before passing to the env model:", state.shape)
            raise ValueError("Invalid state dimension. Expected shape: (3, 7, 7)")

        # Flatten the state tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        flattened_state = torch.reshape(state , (state.shape[0], -1))

        # compute the model free part
        #model_free_hidden = self.model_free(state)
        output_state_representation = self.model_free_agent(flattened_state)

=======
    def forward(self, state1, state2, action_space2):

        # def one_hot(action_index, num_actions=4):
        #     action_one_hot = np.zeros(num_actions)
        #     action_one_hot[action_index] = 1
        #     return action_one_hot
        
        def one_hot(action_indices, num_actions=4):
            # Ensure action_indices are within bounds
            action_indices = action_indices.long() % num_actions
            
            # Create one-hot vectors for each element in the batch
            batch_size = action_indices.size(0)
            action_one_hots = torch.zeros(batch_size, num_actions)
            action_indices = action_indices.view(batch_size, 1)  # Ensure indices have shape (batch_size, 1)
            action_one_hots.scatter_(1, action_indices, 1)
    
            return action_one_hots
        
        action_space2 = one_hot(action_space2)
        flattened_state = state1
>>>>>>> Stashed changes
        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action
<<<<<<< Updated upstream
            # TODO: add exploration
            action1 = self.distilledpolicy(flattened_state)
            action2 = other_agent_distilled_policies(flattened_state)
            

            # Pass the concatenated state-action to the environment model
            next_state = self.env_model(state, action1, action2)
=======

            action1 =self.distilledpolicy(flattened_state)
            

            # Pass the concatenated state-action to the environment model
            
            next_state, _ = self.env_model(state1, state2, action1, action_space2)
>>>>>>> Stashed changes

            # Store the next state in the imagined states
            flattened_state = torch.reshape(next_state, (next_state.size(0), -1))
            imagined_states.append(flattened_state)
            state1 = next_state

        # Concatenate the imagined states along the last dimension 
        imagined_states = torch.cat(imagined_states, dim=0)
       
        
        #encoded_imagined_states = self.imagination(imagined_states)
        encoded_imagined_states = imagined_states
        
        action_prob = self.policy_head(encoded_imagined_states)

        #   Return the action prob 
        return action_prob
    

class I2A_FindTreasure2(nn.Module):
    def __init__(self, state_dim, action_dim, rollout_len=1, hidden_dim=256):
        super(I2A_FindTreasure2, self).__init__()
   
        self.state_dim = state_dim
        self.flattened_state_dim = state_dim[0] * state_dim[1] * state_dim[2]
  
        self.action_dim = action_dim
        self.rollout_len = rollout_len



        # Define the environment model
        self.env_model = EnvironmentModel(self.state_dim,self.state_dim, self.action_dim, self.action_dim)
        
        #Load the pretrained env model
        self.env_model.load_state_dict(torch.load('env_model.pth'))
        
        # Define the imagination module (rollout encoders)
        self.imagination = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # the model free part of I2A
        self.model_free_agent = ModelFreeAgent(self.state_dim, self.action_dim)

        # # Define the policy head

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288, action_dim[0])  # Update the input size for fc1
        

        # Define the policy head
        self.policy_head = nn.Sequential(
            self.conv1,
            self.flatten,
            self.fc1,
            
        )


        #Define the distilled policy
        self.distilledpolicy = DistillPolicyAgent(self.flattened_state_dim, self.action_dim)

    def forward(self, state1, state2, action_space1):
        

        # def one_hot(action_index, num_actions=4):
        #     action_one_hot = np.zeros(num_actions)
        #     action_one_hot[action_index] = 1
        #     return action_one_hot
                
        def one_hot(action_indices, num_actions=4):
            # Ensure action_indices are within bounds
            action_indices = action_indices.long() % num_actions
            
            # Create one-hot vectors for each element in the batch
            batch_size = action_indices.size(0)
            action_one_hots = torch.zeros(batch_size, num_actions)
            action_indices = action_indices.view(batch_size, 1)  # Ensure indices have shape (batch_size, 1)
            action_one_hots.scatter_(1, action_indices, 1)
    
            return action_one_hots
        action_space1 = one_hot(action_space1)
        flattened_state = state2
        # Pass the flattened state to the imagination module
        imagined_states = []
        for _ in range(self.rollout_len):
            #using Distilled policy to select an action

            action2 =self.distilledpolicy(flattened_state)
            #action2 = action2.squeeze(0)

            # Pass the concatenated state-action to the environment model
            
            _ , next_state = self.env_model(state1, state2, action2, action_space1)

            # Store the next state in the imagined states

            flattened_state= next_state
            imagined_states.append(flattened_state)
            state1 = next_state

        # Concatenate the imagined states along the last dimension
        imagined_states = torch.cat(imagined_states, dim=-1)
        encoded_imagined_states = self.imagination(imagined_states)
        
        # Concatenate the model free hidden state and the encoded imagined states
        full_hidden = torch.cat([output_state_representation, encoded_imagined_states], dim=-1)

        action_prob = self.policy_head(full_hidden)
        value = self.value_head(full_hidden)
        #   Return the action prob and the state value
        return action_prob, value