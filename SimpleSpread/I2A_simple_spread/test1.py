from pettingzoo.mpe import simple_spread_v3
from Imagination_Core import I2A_simplespread


from config1 import hyperparameters_agent1, hyperparameters_agent2
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

agent1 = I2A_simplespread(input_size, output_size, hyperparameters_agent1.rollout_len, agent_mode=1)
agent2 = I2A_simplespread(input_size, output_size, hyperparameters_agent2.rollout_len, agent_mode=2)
agent3 = I2A_simplespread(input_size, output_size, hyperparameters_agent1.rollout_len, agent_mode=3)

# Load the state dict of the trained models
agent1.load_state_dict(torch.load('agent_1_model.pth'))
agent2.load_state_dict(torch.load('agent_2_model.pth'))
agent3.load_state_dict(torch.load('agent_2_model.pth'))

# Set the agents in evaluation mode
agent1.eval()
agent2.eval()
agent3.eval()


# Function to select an action using the current policy

def select_action ( model, state1, state2, state3, distilledpolicy1,distilledpolicy2, epsilon):
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
            state3 = torch.tensor(state3, dtype=torch.float32)
            state3 = torch.Tensor(state3).unsqueeze(0)
            action_probs = model(state1 ,state2, state3, distilledpolicy1, distilledpolicy2 )
            #action_probs = torch.Tensor(action_probs).squeeze(1)
            
            action = np.array([int(torch.argmax(action_probs, dim=1).item())], dtype=np.int64)
    return action.item() 
            
max_iter = 1000

total_mean_rewards = []
for i in range(max_iter):
    print("iter= ", i)
    observations, infos = env.reset()
    state1= observations['agent_0']
    state2= observations['agent_1']
    state3= observations['agent_2']
    env.render()


    
    action1 = select_action(agent1, state1, state2, state3, agent2.distilledpolicy, agent3.distilledpolicy, epsilon=0)
    action2 = select_action(agent2, state1, state2, state3, agent1.distilledpolicy, agent3.distilledpolicy, epsilon=0)
    action3 = select_action(agent3, state1, state2, state3, agent1.distilledpolicy, agent2.distilledpolicy, epsilon=0)

    # Execute the actions and store the experiences for both agents
    action_list = {'agent_0':int(action1), 'agent_1': int(action2), 'agent_2':  int(action3)}

    print()
    observations, reward, terminations, truncations, infos = env.step(action_list)
    total_mean_rewards.append(reward)
    if terminations or truncations:
        print('find goal, reward', reward)
        env.reset()
    
