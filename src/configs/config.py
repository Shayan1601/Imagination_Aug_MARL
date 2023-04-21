import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict, List

# the dataclass decorator adds automatically the functions
# __init__ and __repr__ but it requires fields. A field
# is defined as a class variable that has a type annotation.


@dataclass
class I2AConfig:

    # env desc
    env: str
    state_dim: int
    action_dim: int
    max_ep_steps: int = 200

    # memory desc
    capacity: int = int(1e4) 


    # training desc
    num_episodes: int = 30
    train_steps: int = int(5e6)
    batch_size: int = 20
    eval_episode: int = 4
    eval_render: bool = True
    save_every: int = int(1e5) 

    # objective desc
    grad_clip: float = 100.0
    gamma: float = 0.99
    lr: float=1e-3

    # actor critic
    actor_critic: Dict = field(
        default_factory=lambda: {
            "hidden_dim": 2,
            "rollout_len": 5,
            "activation": nn.ReLU,
        }
    )
