# RL-for-ViT



This is the official implementation of the paper: 

## Abstract
In recent years, Artificial Intelligence has experienced extraordinary growth. A lot of this progress is thanks to the introduction of Transformers and Vision Transformers (ViT), which are fundamental for solving Natural Language Processing and Computer Vision tasks respectively. However, one issue with ViTs concerns computational cost, because, starting from an image divided into patches, for each layer, it is necessary to compute the attention of each patch with respect to all others. For this reason, researchers proposed many solutions that try to reduce the cost of attention layers, using techniques like quantization, knowledge distillation, and manipulation of the input images. In this scenario, we propose a new approach named AgentViT to address the computational cost of a ViT. Specifically, we propose a Reinforcement Learning-based solution that aims to train an agent to select the most important patches in order to improve the learning of a ViT. Reducing the number of patches processed by a ViT leads to the reduction of its training time while maintaining competitive performance. We tested AgentViT on CIFAR10 in the image classification task and showed its promising performance against baseline ViTs and similar approaches available in the literature.


## Usage

This repository contains a jupyter notebook which can be downloaded and executed locally, or can be directly runned using Google Colab. In the latter case there is already a cell in which all the necessary libraries are installed. Otherwise the required libraries are displayed in Section [Requirements](#requirements)

In the second cell you need to define the path in which results and models are meant to be saved. The model is trained and tested with CIFAR10 dataset, so the folder where is the CIFAR10 dataset need to be specified; if the indicated folder doesn't contains it, then a copy is automatically downloaded and used for training. 

> [!TIP]
> We suggest to use Colab GPU environment for faster training.

## Parameters 

### Replay Memory Parameters:
```
buffer_size: size of the replay memory;
buffer_batch_size: number of element randomly sampled from the replay memory;
```

### Agent Parameters:
```
gamma: discount factor which causes rewards to lose their value over time;
eps_start: initial value of epsilon;
eps_end: ending value of epsilon;
eps_decay: decay factor of epsilon (the value of epsilon decays exponentially during epochs);

lr: learning rate of the Agent Network;

tau: soft update coefficient;
update_every: how often run the soft update process;

get_reward_every: how often the agent receives a reward and optimizes itself;

time_weight: weights the number of patches selected by the agent;
loss_weight: weights the loss gain compared to the first iteration;

n_patch_selected: how many patches the agent should select;
```

> [!IMPORTANT]
> The agent tries to select a number of patches equal to n_patch_selected. However, in some cases it will select a larger or smaller number based on the batch of input images. The number of patches selected depends on the weight given to the two parameters time_weight and loss_weight.

### ViT Parameters:
```
patch: number of patches;
att_dim: dimension of the attention layers;
epoche = training epochs;
learning_rate = learning rate of the SimpleViT;
```


## Requirements <a name="requirements"></a>
