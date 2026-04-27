# Cooperative Capabilities of Multi-Agent Systems: An AI 'Hunger Games' Simulation


### Project Overview
This research was conducted for my final project. I chose to focus on one of my key interests in Comp Sci which is multi-agent reinforcement learning. This project explores cooperation in this setting using a grid based environment. Multiple agents are also trained using PPO to collect resources whilst interacting with each other. T

The main aim to understand whether cooperative behaviour emerges naturally and how different reward structures influence this behaviour. 


## Aims and Objectives
- Investigate whether cooperation emerges in multi-agent systems
- Compare different reward structures and their effects on behaviour
- Analyse how reward shaping impacts learning and stability
- Produce visualisations and measure performance to track all of this


## Main Artefacts
/agents - contains PPO implementation and communication logic
/env - contains the gridworld environment and rendering logic
/train - contains the training scripts
/ demo - contains the scripts I plan on using in the live demo (viva)
/analysis - contains scripts for generating graphs and metrics
/results - stores output data such as CSV and JSON files 
/checkpoints - stores trained models

I do not plan on committing the last 3 folders to GitLab as they are extremely large and contain a lot of unnecessary data that was generated from testing and running the system loads.


## How to run (reproducibility)

### Setup
Install dependencies:
pip install -r requirements.txt

### Training Agents
Run training for each reward scheme:
python3 train/train_headless.py --num-episodes 50000 --reward-scheme selfish
python3 train/train_headless.py --num-episodes 50000 --reward-scheme cooperative
python3 train/train_headless.py --num-episodes 50000 --reward-scheme mixed

### Analysis
python3 analysis/run_analysis.py