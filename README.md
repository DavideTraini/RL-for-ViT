# RL-for-ViT



This is the official implementation of the paper: Adaptive Patch Selection to Improve Vision Transformers through Reinforcement Learning

## Abstract
In recent years, Transformers have been revolutionizing the management of Natural Language Processing tasks, and Vision Transformers (ViTs) promise to do the same for Computer Vision ones. However, the adoption of ViTs is held back by their computational cost. Indeed, starting from an image divided into patches, for each layer it is necessary to compute the attention of each patch with respect to all the others. Researchers have proposed many solutions that aim to reduce the computational cost of attention layers by adopting techniques such as quantization, knowledge distillation and manipulation of input images. In this paper, we aim to make a contribution to address this issue. In particular, we propose a new framework, called AgentViT, which uses Reinforcement Learning for training an agent that selects the most important patches for the learning of a ViT. The goal of AgentViT is to reduce the number of patches processed by a ViT, and thus its training time, while still maintaining competitive performance. We tested AgentViT on CIFAR10 in the image classification task and obtained promising performance if compared to baseline ViTs and other related approaches available in the literature.

## Usage

This repository contains a jupyter notebook which can be downloaded and executed locally, or can be directly runned using Google Colab. In the latter case there is already a cell in which all the necessary libraries are installed. Otherwise the required libraries are displayed in Section [Requirements](#requirements)

In the second cell you need to define the path in which results and models are meant to be saved. The model is trained and tested with CIFAR10 dataset, so the folder where is the CIFAR10 dataset need to be specified; if the indicated folder doesn't contains it, then a copy is automatically downloaded and used for training. 

> [!TIP]
> We suggest to use Colab GPU environment for faster training.

## Parameters 


### Replay Memory Parameters:
**`buffer_size`**: size of the replay memory;  
**`buffer_batch_size`**: number of element randomly sampled from the replay memory;


### Agent Parameters:
**`gamma`**: discount factor which causes rewards to lose their value over time;  
**`eps_start`**: initial value of epsilon;  
**`eps_end`**: ending value of epsilon;  
**`eps_decay`**: decay factor of epsilon (the value of epsilon decays exponentially during epochs);  

**`eta`**: learning rate of the Agent Network;  

**`tau`**: soft update coefficient;  
**`update_every`**: how often run the soft update process;  

**`frequency`**: how often the agent receives a reward and optimizes itself;  

**`alpha`**: weight of the reward related to training loss against the number of patches selected;   

**`n_patch_desired`**: how many patches the agent should select;  

> [!IMPORTANT]
> The agent tries to select a number of patches equal to n_patch_selected. However, in some cases it will select a larger or smaller number based on the batch of input images. The number of patches selected depends on the weight given to the two parameters time_weight and loss_weight.

### ViT Parameters:
**`patch_size`**: dimension of patches;  
**`att_dim`**: dimension of the embedding;  
**`depth`**: number of transformer layers;  
**`heads`**: number of multi attention heads;  
**`mlp_dim`**: dimension of the multi layer perceptron layer;  
**`epoche`**: training epochs;  
**`learning_rate`**: learning rate of the SimpleViT;  


## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
torch==2.1.0  
torchvision=0.16.0  
sklearn=1.2.2  
einops=0.7.0  
gym=0.25.2  
numpy=1.23.5  
pandas=1.5.3  
matplotlib=3.7.1  
```
