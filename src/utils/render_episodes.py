import gymnasium as gym
import argparse
from gymnasium.wrappers.human_rendering import HumanRendering



def render_episodes(env_name:str, num_steps:int, model= None):
    env = gym.make(env_name,render_mode="rgb_array")
    env = HumanRendering(env)
    env.reset()
    for _ in range(num_steps):
        if model == None:
            action = env.action_space.sample()
        # implement the model here
        else:
            continue
        _, _, _, _,_ = env.step(action)
        env.render()
    env.close()


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="Human Rendering of the I2A agent")
    # Training settings (just the most important ones, the complete configuration is in config.py)
    parser.add_argument("--num_steps", type=int, default=100, help="number of steps to execute")
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="Name of the environment")
    args = parser.parse_args()
    render_episodes(args.env_name,args.num_steps)