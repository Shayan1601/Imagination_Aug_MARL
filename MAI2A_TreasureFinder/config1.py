#config file containing the hyper-parameters


config = {
    "conv1_out_channels": 32,
    "conv1_filter_size": 3,
    "conv1_stride": 1,
    "conv2_out_channels": 16,
    "conv2_filter_size": 3,
    "conv2_stride": 1,
    "fc1_out_dim": 32
}


#hyper-parameters

class Hyperparameters:
    def __init__(self, num_episodes, batch_size, replay_memory_size, rollout_len, gamma, lr,
                 world_loss_weight, distil_policy_loss_weight):
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lr = lr
        self.world_loss_weight = world_loss_weight
        self.distil_policy_loss_weight = distil_policy_loss_weight
        # Optionally, if we want to train in separate steps, we can add these two parameters
        # self.world_model_training_steps = world_model_training_steps
        # self.distil_policy_training_steps= distil_policy_training_steps


# Define your hyperparameters here
hyperparameters_agent1 = Hyperparameters(
    num_episodes=2000,
    batch_size=20,
    replay_memory_size=10000,
    rollout_len=5,
    gamma=0.99,
    lr=1e-3,
    world_loss_weight = 0.5,
    distil_policy_loss_weight=0.5
)

hyperparameters_agent2 = Hyperparameters(
    num_episodes=2000,
    batch_size=20,
    replay_memory_size=10000,
    rollout_len=5,
    gamma=0.99,
    lr=1e-3,
    world_loss_weight = 0.5,
    distil_policy_loss_weight=0.5
)
