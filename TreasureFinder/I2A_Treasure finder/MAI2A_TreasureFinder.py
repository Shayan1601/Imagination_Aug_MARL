#Training I2A agents for the Treasure Finder multi agent environment
#importing the dependencies
#I'm trying to pretrain and deploy the env model on this one
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.chdir('/Users/shayan/Desktop/Reboot treasure/March')

#from Treasure_Finder_gymformat import TreasureFinderEnv
from env_FindTreasure import EnvFindTreasure
from Imagination_Core import I2A_FindTreasure

from config1 import hyperparameters_agent1, hyperparameters_agent2



# Defining the replay memory for both agents
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        return len(self.memory)

# Creating replay memory for both agents
replay_memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)
replay_memory_agent2 = ReplayMemory(hyperparameters_agent2.replay_memory_size)

# Function to select an action using the current policy
def select_action ( model, state1, state2, distilledpolicy, epsilon):
    #if epsilon > 0.95:
    if np.random.rand() < epsilon:
        # Random action
        action = torch.randint(0, 4, (1,))
    else:
        # Use I2A module to produce action
        with torch.no_grad():
            #action_space = torch.tensor([output_size[0]], dtype=torch.float32).unsqueeze(0)
            state1 = torch.tensor(state1, dtype=torch.float32)
            state1 = torch.Tensor(state1).unsqueeze(0)
            state2 = torch.tensor(state2, dtype=torch.float32)
            state2 = torch.Tensor(state2).unsqueeze(0)
            action_probs = model(state1 ,state2, distilledpolicy)
            #action_probs = torch.Tensor(action_probs).squeeze(1)
            
            action = np.array([int(torch.argmax(action_probs, dim=1).item())], dtype=np.int64)
    return action.item() 

# Main function to train and test the I2A agent
if __name__ == "__main__":
    # Create the environment
    env = EnvFindTreasure(7)

    input_size = (3, 3, 3)  # Update with the appropriate state size attribute
    output_size = (4,)
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.001
    max_time_steps = 50  # Maximum number of time steps
    update_target_frequency = 7


    # Instantiate the I2A models and optimizers for both agents
    model_agent1 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent1.rollout_len, agent_mode=1)
    model_agent2 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent2.rollout_len, agent_mode=2)
    
    optimizer_agent1 = optim.Adam(model_agent1.policy_head.parameters(), lr=hyperparameters_agent1.lr)
    optimizer_agent2 = optim.Adam(model_agent2.policy_head.parameters(), lr=hyperparameters_agent2.lr)
    optimizer_world_model_agent1 = optim.Adam(model_agent1.env_model.parameters(), lr=hyperparameters_agent1.lr)
    optimizer_world_model_agent2 = optim.Adam(model_agent2.env_model.parameters(), lr=hyperparameters_agent2.lr)
    optimizer_distilled_policy_agent1 = optim.Adam(model_agent1.distilledpolicy.parameters(), lr=hyperparameters_agent1.lr)
    optimizer_distilled_policy_agent2 = optim.Adam(model_agent2.distilledpolicy.parameters(), lr=hyperparameters_agent2.lr)
   
    world_model_loss_function = nn.CrossEntropyLoss()
    distil_policy_loss_function = nn.NLLLoss()
    # Initialize the replay memory for both agents
    memory_agent1 = ReplayMemory(hyperparameters_agent1.replay_memory_size)
    memory_agent2 = ReplayMemory(hyperparameters_agent2.replay_memory_size)
    
    #Target network for improving the training of the agents
    target_network1 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent1.rollout_len, agent_mode=1)
    target_network2 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent2.rollout_len, agent_mode=2)
    
    
    # Main training loop

    mean_rewards_agent1 = []  # To store mean rewards for Agent 1
    mean_rewards_agent2 = []  # To store mean rewards for Agent 2
    mean_R_Plot = []
    
    # Start time
    start_time = time.time()

    for episode in range(hyperparameters_agent1.num_episodes):
        state = env.reset()

        state1 = env.get_agt1_obs()
        state2 = env.get_agt2_obs()
        
        done = False
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0


        for t in count():
            # Select actions based on the current policies for both agents

            action_agent1 = select_action(model_agent1, state1, state2, model_agent2.distilledpolicy,epsilon)
            action_agent2 = select_action(model_agent2, state1, state2, model_agent1.distilledpolicy,epsilon)

            # Execute the actions and store the experiences for both agents
            action_list= [action_agent1, action_agent2]
            reward, done = env.step(action_list)
            next_state1 = env.get_agt1_obs()
            next_state2 = env.get_agt2_obs()

        

            memory_agent1.push(state1, action_agent1, reward, next_state1, done)
            memory_agent2.push(state2, action_agent2, reward, next_state2, done)

            ###############
            # TRAIN AGENT #
            ###############

            if len(memory_agent1) >= hyperparameters_agent1.batch_size:

                states1, actions1, rewards1, next_states1, dones = memory_agent1.sample(hyperparameters_agent1.batch_size)
                states2, actions2, rewards2, next_states2, dones = memory_agent2.sample(hyperparameters_agent2.batch_size)
     
                Trajectory = [states1, actions1, rewards1, next_states1]


                states1 = torch.Tensor(states1)
                states2 = torch.Tensor(states2)
                next_states1= torch.Tensor(next_states1)
                next_states2= torch.Tensor(next_states2)
                actions1 = torch.LongTensor(actions1)
                actions2 = torch.LongTensor(actions2)
                rewards1 = torch.Tensor(rewards1)
                rewards2 = torch.Tensor(rewards2)
                dones = torch.Tensor(dones)
 
                
                # Compute the current Q values for both agents
                
                action_probs_agent1= model_agent1(states1,states2, model_agent2.distilledpolicy)
                actions11 = actions1.unsqueeze(-1)
                      
                action_values_agent1 = torch.gather(action_probs_agent1, 1,actions11)
                               
                next_q_values1 = target_network1(next_states1, next_states2, model_agent2.distilledpolicy).max(1)[0].detach()
                target_q_values1 = rewards1 + hyperparameters_agent1.gamma * next_q_values1 * (1 - dones)
                target_q_values1 = target_q_values1.unsqueeze(1)
                

                action_probs_agent2 = model_agent2(states1, states2, model_agent1.distilledpolicy)
                actions22 = actions2.unsqueeze(-1)
                                    
                action_values_agent2 = torch.gather(action_probs_agent2, 1, actions22)
                
                next_q_values2 = target_network2(next_states1, next_states2, model_agent1.distilledpolicy).max(1)[0].detach()
                target_q_values2 = rewards2 + hyperparameters_agent2.gamma * next_q_values2 * (1 - dones)
                target_q_values2 = target_q_values2.unsqueeze(1)
                
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
                
                actions111 = one_hot_encode(actions11)
                actions222 = one_hot_encode(actions22)
                action_probs_agent11= one_hot(action_probs_agent1)
                action_probs_agent22= one_hot(action_probs_agent2)
                # WORLD MODEL LOSS #
                imagined_states1, imagined_states2 = model_agent1.env_model(states1, states2, actions111, actions222)
                world_loss_agent1 =F.mse_loss(imagined_states1, next_states1) 
                imagined_states1, imagined_states2 = model_agent2.env_model(states1, states2, actions111, actions222)
                world_loss_agent2 =F.mse_loss(imagined_states2, next_states2) 
                
               # DISTIL POLICY LOSS # 
                dist_actions1 = model_agent1.distilledpolicy(states1)
                dist_actions2 = model_agent1.distilledpolicy(states2)
                distilled_policy_loss_agent1 = F.mse_loss(dist_actions1, action_probs_agent11)
                distilled_policy_loss_agent2 = F.mse_loss(dist_actions2, action_probs_agent22)
                
                # Compute the loss for both agents
                loss_agent1 = nn.functional.smooth_l1_loss(action_values_agent1, target_q_values1)
                loss_agent2 = nn.functional.smooth_l1_loss(action_values_agent2, target_q_values2)
                
                total_loss_agent1 = loss_agent1 + hyperparameters_agent1.world_loss_weight * world_loss_agent1 + \
                                    hyperparameters_agent1.distil_policy_loss_weight * distilled_policy_loss_agent1
                
                total_loss_agent2 = loss_agent2 + hyperparameters_agent2.world_loss_weight* world_loss_agent2 + \
                                    hyperparameters_agent2.distil_policy_loss_weight* distilled_policy_loss_agent2


    
                
                # Backpropagation and optimization
                    # Set requires_grad to False for all parameters in the other agent's distilled policy and env model
                for param in model_agent2.distilledpolicy.parameters():
                    param.requires_grad = False
                for param in model_agent1.distilledpolicy.parameters():
                    param.requires_grad = False
                
                for param in model_agent1.env_model.parameters():
                    param.requires_grad = False
                
                for param in model_agent2.env_model.parameters():
                    param.requires_grad = False
                     
                # for param in model_agent1.encoder.parameters():
                #     param.requires_grad = False
                
                # for param in model_agent2.encoder.parameters():
                #     param.requires_grad = False
                
                #updating each network seperately
                
                optimizer_distilled_policy_agent2.zero_grad()
                distilled_policy_loss_agent2.backward()
                optimizer_distilled_policy_agent2.step()
                
                optimizer_agent2.zero_grad()
                loss_agent2.backward()
                optimizer_agent2.step()

                
                optimizer_world_model_agent2.zero_grad()
                world_loss_agent2.backward()
                optimizer_world_model_agent2.step()

                #updating each network seperately
                
                optimizer_distilled_policy_agent1.zero_grad()
                distilled_policy_loss_agent1.backward()
                optimizer_distilled_policy_agent1.step()
                
                optimizer_agent1.zero_grad()
                loss_agent1.backward()
                optimizer_agent1.step()
                
                optimizer_world_model_agent1.zero_grad()
                world_loss_agent1.backward()
                optimizer_world_model_agent1.step()

                
            # Update the states and episode rewards for both agents
            state1 = next_state1
            state2 = next_state2
            
            episode_reward_agent1 += reward
            episode_reward_agent2 += reward
       
            if done or t >= max_time_steps:
                break
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        #updating the target networks
        if episode % update_target_frequency == 0:
            target_network1.load_state_dict(model_agent1.state_dict())
            target_network2.load_state_dict(model_agent2.state_dict())


        # Calculate episode and mean rewards 
        mean_rewards_agent1.append(episode_reward_agent1)
        mean_rewards_agent2.append(episode_reward_agent2)

        #print(f"Episode {episode+1}:  Reward = {episode_reward_agent1}")
        
        mean_reward_agent1 = sum(mean_rewards_agent1) / len(mean_rewards_agent1)
        mean_R_Plot. append(mean_reward_agent1)
        if (episode + 1) % 100 == 0:
            
            print(f"Episode {episode+1}: Mean Reward = {mean_reward_agent1}")
            
    # Calculate total mean rewards
    total_mean_reward_agent1 = sum(mean_rewards_agent1) / len(mean_rewards_agent1) 
    # Print total mean rewards
    print(f"Total Mean Reward after {hyperparameters_agent1.num_episodes} episodes : {total_mean_reward_agent1}") 
    
    # End time
    end_time = time.time()
    # Total training time
    training_time_seconds = end_time - start_time
    training_hours = int(training_time_seconds // 3600)
    training_minutes = int((training_time_seconds % 3600) // 60)

    print(f"Total training time: {training_hours} hours and {training_minutes} minutes")
    training_time_str = f"Training Time: {training_hours} hours {training_minutes} minutes"

    
    # Plotting
    episodes = list(range(1, hyperparameters_agent1.num_episodes + 1, 1))
    plt.plot(episodes, mean_R_Plot, label='Mean Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Episodes')
    plt.grid(True)
    plt.legend()
    # Display training time beneath the y-axis
    plt.text(20, min(mean_R_Plot) - 0.2, training_time_str, ha='left')
    plt.show()
          
    # Save the agents parameters after training
    torch.save(model_agent1.state_dict(), f'agent_{1}_model.pth') 
    torch.save(model_agent2.state_dict(), f'agent_{2}_model.pth')
    torch.save(model_agent1.distilledpolicy.state_dict(), f'agent_{1}_distilledp.pth') 
    torch.save(model_agent2.distilledpolicy.state_dict(), f'agent_{2}_distilledp.pth')  
    
       
            
            
            




       



    