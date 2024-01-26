import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TRAIN import MADQNAgent
from pettingzoo.mpe import simple_spread_v3
import torch
import numpy as np

# Create the environment
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=500, continuous_actions=False)

input_size = (18,)  # Update with the appropriate state size attribute
output_size = (5,) #env.action_size  # Update with the appropriate action size attribute
buffer_size = 100000
batch_size = 120
discount_factor = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.001
update_target_frequency = 5

# Load trained agent models
agent1 = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)  # Replace YourAgentClass with the actual class of your agent
agent2 = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)
agent3 = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)
# Load the state dict of the trained models
agent1.q_network.load_state_dict(torch.load('agent_0_model.pth'))
agent2.q_network.load_state_dict(torch.load('agent_1_model.pth'))
agent3.q_network.load_state_dict(torch.load('agent_2_model.pth'))

# Set the agents in evaluation mode
agent1.q_network.eval()
agent2.q_network.eval()
agent3.q_network.eval()

max_iter = 1000
total_mean_rewards = []

for i in range(max_iter):
    print("iter= ", i)
    observations, infos = env.reset()
    state1= observations['agent_0']
    state2= observations['agent_1']
    state3= observations['agent_2']
    env.render()
    
    action1 = agent1.get_action(state1, epsilon)
    action2 = agent2.get_action(state2, epsilon)
    action3 = agent3.get_action(state3, epsilon)
    action_list = {'agent_0':int(action1), 'agent_1': int(action2), 'agent_2':  int(action3)}
 
    print()
    observations, reward, terminations, truncations, infos = env.step(action_list)
    
    if terminations or truncations:
        print('Episode Ended, reward', reward)
        env.reset()
        
    total_mean_rewards.append(reward['agent_0'])

# Calculate and print the total mean rewards at the end
if total_mean_rewards:
    total_mean_reward = sum(total_mean_rewards) / len(total_mean_rewards)
    print(f'Total Mean Reward over {len(total_mean_rewards)} iterations: {total_mean_reward}')
else:
    print('No episodes completed.')