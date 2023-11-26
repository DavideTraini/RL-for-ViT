# RL-for-ViT



This is the official implementation of the paper: 

## Abstract



## Usage

This repository contains a jupyter notebook which can be downloaded and executed locally, or can be directly runned using Google Colab. In the latter case there is already a cell in which all the necessary libraries are installed. Otherwise the required libraries are displayed in Section [Requirements](#requirements)

In the second cell you need to define the path in which results and models are meant to be saved. The model is trained and tested with CIFAR10 dataset, so the folder where is the CIFAR10 dataset need to be specified; if the indicated folder doesn't contains it, then a copy is automatically downloaded and used for training. 

> [!TIP]
> We suggest to use Colab GPU environment for faster training

## Parameters 

Replay Memory Parameters
```
buffer_size: size of the replay memory;
buffer_batch_size: number of element randomly sampled from the replay memory;
```
Agent Parameters:
```
gamma: discount factor which causes rewards to lose their value over time;
eps_start: initial value of \(\varepsilon\)
eps_end: ending value of \(\varepsilon\)
eps_decay: decay factor of \(\varepsilon\) (the value of epsilon decays exponentially during epochs)

lr: learning rate of the Agent Network

tau: soft update coefficient
update_every: how often run the soft update process

get_reward_every: how often the agent receives a reward and optimizes itself

time_weight: weights the number of patches selected by the agent;
loss_weight: weights the loss gain compared to the first iteration;

n_patch_selected: how many patches the agent should select;
```

> [!IMPORTANT]
> The agent tries to select a number of patches equal to n_patch_selected. However, in some cases it will select a larger or smaller number based on the batch of input images. The number of patches selected depends on the weight given to the two parameters time_weight and loss_weight.

ViT Parameters:
```
patch: number of patches 
att_dim: dimension of the attention layers
epoche = training epochs
learning_rate = learning rate of the SimpleViT
```


## Requirements <a name="requirements"></a>
