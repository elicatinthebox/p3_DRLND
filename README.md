[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Deep Reinforcement Learning Nanodegree - Project 3: Multi-Agent Collaboration & Competition

### Introduction

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Getting Started


- Configure a Python 3.6 / PyTorch 0.4.0 environment according to the requirements described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) and clone the Udacity's repository.
- Install [Jupyter Notebook](https://jupyter.org/)

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip 

Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip 

Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip

Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip 

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment

2. Place the file in the root folder of the project, and unzip (or decompress) the file. 
3. Open the project folder inside a terminal
4. Run the Jupyter notebook named `Tennis.ipynb` using the command _jupyter notebook_
5. Run the cells inside the Jupyter Notebook named `Tennis.ipynb`




### Instructions

To know more about how I've devloped the project please take a look at the [REPORT.md](https://github.com/elisaromondia/p3_DRLND/blob/master/REPORT.md) document and run the code provided in the [Tennis.ipynb](https://github.com/elisaromondia/p3_DRLND/blob/master/Tennis.ipynb) Jupyter Notebook.

