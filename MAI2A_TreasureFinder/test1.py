from env_FindTreasure import EnvFindTreasure
from Imagination_Core import I2A_FindTreasure

from config1 import hyperparameters_agent1, hyperparameters_agent2
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
agent1 = I2A_FindTreasure( input_size, output_size, hyperparameters_agent1.rollout_len)  # Replace YourAgentClass with the actual class of your agent
agent2 = I2A_FindTreasure( input_size, output_size, hyperparameters_agent2.rollout_len)

# Load the state dict of the trained models
agent1.load_state_dict(torch.load('agent_1_model.pth'))
agent2.load_state_dict(torch.load('agent_2_model.pth'))

# Set the agents in evaluation mode
agent1.eval()
agent2.eval()

# Function to select an action using the current policy
def select_action(model, state, output_size, epsilon):
    #if epsilon > 0.95:
    if np.random.rand() < epsilon:
        # Random action
        action = torch.randint(0, output_size[0], (1,))
    else:
        # Use I2A module to produce action
        with torch.no_grad():
            action_space = torch.tensor([output_size[0]], dtype=torch.float32).unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float32)
            state = torch.Tensor(state).unsqueeze(0)
            action_probs = model(state, action_space)
            action_probs = torch.Tensor(action_probs).squeeze(1)
            
            action = np.array([int(torch.argmax(action_probs, dim=1).item())], dtype=np.int64)
    return action.item() 
            
max_iter = 1000

total_mean_rewards = []
for i in range(max_iter):
    print("iter= ", i)
    env.render()

    # Replace random actions with actions predicted by the agents
    action1 = select_action(agent1, env.get_agt1_obs(),(4,),  epsilon=0)  # Set epsilon to 0 for greedy actions
    action2 = select_action(agent2, env.get_agt2_obs(),(4,), epsilon=0)
    action_list = [action1, action2]

    print()
    reward, done = env.step(action_list)
    total_mean_rewards.append(reward)
    if done:
        print('find goal, reward', reward)
        env.reset()
    
