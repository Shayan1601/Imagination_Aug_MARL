# Imagination_Aug_MARL
In this project we want to extend the Imagination augmented model-based reinforcement learning to multi-agent scenarios and see how it improves
the training process in terms of performance, robustness and sample efficinecy.
We are building on the works done by researchers from DeepMind. The original paper, titled "Imagination-Augmented Agents for Deep Reinforcement Learning," can be found here: https://arxiv.org/abs/1707.06203.


# Anaconda Installation

Download Anaconda for python3 and in a new terminal use the following code
```
conda create -n img_aug_marl python=3.9
conda activate img_aug_marl
```

Install the following packages with pip
```
pip install torch torchvision torchaudio
pip install wandb
pip install gymnasium
pip install "gymnasium[box2d]"
```


# Directory Structure
```
├── src
│   ├── agents
│   │   ├── i2a_sa.py      
│   │   ├── i2a_ma.py
│   ├── configs
│   ├── envs
│   ├── logs
│   ├── models
│   ├── utils
├── LICENSE
├── README.md 
├──  
└── .gitignore
```

