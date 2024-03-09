from env_FindTreasure import EnvFindTreasure
from Imagination_Core import I2A_FindTreasure
import os

from config1 import hyperparameters_agent1, hyperparameters_agent2
import torch
import numpy as np
os.chdir('/Users/shayan/Desktop/Reboot treasure/I2A-a2c-T8')

# Create the environment
env = EnvFindTreasure(9)

input_size = (3,3,3)  # Update with the appropriate state size attribute
output_size = (4,) #env.action_size  # Update with the appropriate action size attribute
buffer_size = 1
batch_size = 1
discount_factor = 0.99
learning_rate = 0.0001

# Load trained agent models
agent1 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent1.rollout_len, agent_mode=1)  # Replace YourAgentClass with the actual class of your agent
agent2 = I2A_FindTreasure(input_size, output_size, hyperparameters_agent2.rollout_len, agent_mode=2)

# Load the state dict of the trained models
agent1.load_state_dict(torch.load('agent_1_model.pth'))
agent2.load_state_dict(torch.load('agent_2_model.pth'))

# Set the agents in evaluation mode
agent1.eval()
agent2.eval()


# Function to select an action using the current policy
def select_action ( model, state1, state2, output_size, epsilon):
    #if epsilon > 0.95:
    if np.random.rand() < epsilon:
        # Random action
        action = torch.randint(0, 4, (1,))
    else:
        # Use I2A module to produce action
        with torch.no_grad():
            action_space = torch.tensor([output_size]).unsqueeze(0)
            state1 = torch.tensor(state1, dtype=torch.float32)
            state1 = torch.Tensor(state1).unsqueeze(0)
            state2 = torch.tensor(state2, dtype=torch.float32)
            state2 = torch.Tensor(state2).unsqueeze(0)
            action_probs, _ = model(state1 ,state2, action_space)
            #action_probs = torch.Tensor(action_probs).squeeze(1)
            
            action = np.array([int(torch.argmax(action_probs, dim=1).item())], dtype=np.int64)
    return action.item()
            
max_iter = 1000
action_agent11= torch.randint(0, 4, (1,))
action_agent22= torch.randint(0, 4, (1,))

total_mean_rewards = []
for i in range(max_iter):
    print("iter= ", i)
    env.render()

    # Replace random actions with actions predicted by the agents
    action1 = select_action(agent1, env.get_agt1_obs(),env.get_agt2_obs(), action_agent22,  epsilon=0)  # Set epsilon to 0 for greedy actions
    action2 = select_action(agent2, env.get_agt1_obs(), env.get_agt2_obs(),action_agent11, epsilon=0)
    action_list = [action1, action2]
    # action_agent11= action1
    # action_agent22= action2

    print()
    reward, done = env.step(action_list)
    total_mean_rewards.append(reward)
    if done:
        print('find goal, reward', reward)
        env.reset()
    
