from env_FindTreasure import EnvFindTreasure
from MADQN_TreasureFinder import MADQNAgent
import torch
import numpy as np

# Create the environment
env = EnvFindTreasure(7)

input_size = (3,3,3)  # Update with the appropriate state size attribute
output_size = (4,) #env.action_size  # Update with the appropriate action size attribute
buffer_size = 100000
batch_size = 100
discount_factor = 0.99
learning_rate = 0.001

# Load trained agent models
agent1 = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)  # Replace YourAgentClass with the actual class of your agent
agent2 = MADQNAgent( input_size, output_size, buffer_size, batch_size, discount_factor, learning_rate)

# Load the state dict of the trained models
agent1.q_network.load_state_dict(torch.load('agent_0_model.pth'))
agent2.q_network.load_state_dict(torch.load('agent_1_model.pth'))

# Set the agents in evaluation mode
agent1.q_network.eval()
agent2.q_network.eval()

max_iter = 1000
total_mean_rewards = []

for i in range(max_iter):
    print("iter= ", i)
    env.render()

    # Replace random actions with actions predicted by the agents
    action1 = agent1.get_action(env.get_agt1_obs(), epsilon=0)  # Set epsilon to 0 for greedy actions
    action2 = agent2.get_action(env.get_agt2_obs(), epsilon=0)
    action_list = [action1, action2]

    print()
    reward, done = env.step(action_list)
    
    if done:
        print('find goal, reward', reward)
        env.reset()
        
    total_mean_rewards.append(reward)

# Calculate and print the total mean rewards at the end
if total_mean_rewards:
    total_mean_reward = sum(total_mean_rewards) / len(total_mean_rewards)
    print(f'Total Mean Reward over {len(total_mean_rewards)} iterations: {total_mean_reward}')
else:
    print('No episodes completed.')