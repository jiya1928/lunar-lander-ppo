# LUNARIS - Autonomous Lunar Landing System

This is NOT a game. This is an autonomous control system trained to land a spacecraft using Reinforcement Learning.

The agent learns to land by itself - no human control, no hard-coded rules. Just trial, error and reward.

## Demo

Watch the trained agent land autonomously.
Video: 5 consecutive autonomous landings, 0 crashes
https://github.com/user-attachments/assets/f3ae6ab8-0873-4bab-814d-eb7df13e3140

## The Problem

How do you land a spacecraft on the Moon without a pilot?

The lander starts in the air with random velocity. It has to:
- Fire its main and side thrusters at the right time
- Control its speed and angle
- Land gently between the two flags
- Save as much fuel as possible

## Method

- Algorithm: PPO (Proximal Policy Optimization)
- Framework: Stable-Baselines3 + Gymnasium
- Environment: LunarLander-v3
- Training: 800K+ steps

Reward System:
- +100 for safe landing
- -100 for crash
- -0.3 per frame for main engine fuel
- +10 for each leg touching ground

## Results

Evaluated on 5 unseen episodes:

Episode 1 - Reward 121.31 - Landed
Episode 2 - Reward 288.61 - Landed
Episode 3 - Reward 271.36 - Landed
Episode 4 - Reward 283.40 - Landed
Episode 5 - Reward 131.78 - Landed

Average Reward: 219.29
Success Rate: 5/5 (100%)

A reward above 200 is considered solved in this benchmark.

## Project Structure

LUNARIS/
- train.py - Train the PPO agent
- watch.py - Watch the trained agent land
- evaluate.py - Evaluate performance
- record.py - Record landing videos
- models/ - Trained PPO model

## How to Run

pip install -r requirements.txt
python watch.py

## Future Work

- Add wind / disturbance to test robustness
- Try SAC algorithm for comparison
- Transfer to a custom 3D lunar environment

Built with Python, Gymnasium & Stable-Baselines3.
