import torch
import torch.optim as optim
import numpy as np
import argparse
import sys
from itertools import count
from collections import namedtuple
from src.envs.env_wrapper import EnvWrapper
from src.configs.config import I2AConfig
from src.agents.replay_memory import ReplayMemory
from src.models.actorcritic_model import ActorCriticModel


# Define the namedtuple to store experiences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# function to select an action using the current policy
def select_action(model, state, action_space):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs, _ = model(state, action_space)
    m = torch.distributions.Categorical(action_probs)
    return m.sample().item()


# Main function to train and test the I2A agent
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated: ", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
        print("Cached:    ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")

    parser = argparse.ArgumentParser(description="Imagination-Augmented Agents for Deep Reinforcement Learning")

    # Training settings (just the most important ones, the complete configuration is in config.py)
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="Name of the environment")
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("--rollout_len", type=int, default=5, help="Length of the rollout for imagination")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")

    if sys.argv[0].endswith("ipykernel_launcher.py"):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()


    # Create the environment
    env = EnvWrapper()
    state = env.reset()

    config = I2AConfig(
        env=args.env_name,
        state_dim= env.observation_space.shape[0],
        action_dim=env.action_space.n,
        num_episodes=args.num_episodes
    )

    # Instantiate the I2A model and optimizer
    model = ActorCriticModel(config.state_dim, config.action_dim, **config.actor_critic)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Initialize the replay memory
    memory = ReplayMemory(config.capacity)

    # Main training loop
    for episode in range(config.num_episodes):
        print(f"Episode {episode}/{config.num_episodes}")
        state = env.reset()
        episode_reward = 0

        for t in count():
            if t > config.max_ep_steps:
                break
            # Select an action based on the current policy
            action = select_action(model, state, env.action_space)


            # Execute the action and store the experience
            next_state, reward, done= env.step(action)[0:3]
            memory.push(state, action, reward, next_state, done)

            # Update the state and episode reward
            state = next_state
            episode_reward += reward

            # If enough experiences are collected, perform a training step
            if len(memory) >= config.batch_size:
                experiences = memory.sample(config.batch_size)
                batch = Experience(*zip(*experiences))

               
                # Prepare the data for training
                states = torch.tensor(np.array(batch.state), dtype=torch.float32)
                actions = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
                dones = torch.tensor(np.array(batch.done), dtype=torch.float32).unsqueeze(1)


                # Compute the current Q values
                action_probs, state_values = model(states, env.action_space)
                action_values = action_probs.gather(1, actions)

                # Compute the target Q values
                _, next_state_values = model(next_states, env.action_space)

                target_action_values = rewards + (config.gamma * next_state_values * (1 - dones))

                # Compute the loss and perform a training step
                loss = (action_values - target_action_values.detach()).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Check if the episode is finished
            if done:
                print("Episode: {}, Reward: {}, Timesteps: {}".format(episode, episode_reward, t + 1))
                break

    # Testing the trained agent
    print("Testing the trained agent...")
    test_episodes = 10
    test_rewards = []

    for episode in range(test_episodes):
        state = env.reset()
        episode_reward = 0

        for t in count():
            action = select_action(model, state, env.action_space)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                print("Test Episode: {}, Reward: {}, Timesteps: {}".format(episode, episode_reward, t + 1))
                test_rewards.append(episode_reward)
                break

    print("Average test reward: {:.2f}".format(sum(test_rewards) / test_episodes))
    env.close()


if __name__ == "__main__":
    main()

