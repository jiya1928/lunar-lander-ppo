import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import sys
from rewards import NormalReward, TimePenaltyReward, HackedReward, FixedReward

# Usage: python train.py normal / time / hacked / fixed
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

print(f"Training MODE: {mode} | STEPS: {steps}")

def make_env():
    env = gym.make("LunarLander-v3")
    if mode == "time":
        env = TimePenaltyReward(env)
    elif mode == "hacked":
        env = HackedReward(env)
    elif mode == "fixed":
        env = FixedReward(env)
    else:
        env = NormalReward(env)
    env = Monitor(env)
    return env

env = make_env()

# Eval ke liye alag env, bina wrapper ke - taaki asli score pata chale
eval_env = gym.make("LunarLander-v3")
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{mode}/",
    log_path=f"./logs/{mode}/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10)

model.learn(total_timesteps=steps, callback=eval_callback)
model.save(f"models/{mode}_final")
print(f"DONE! Model saved: models/{mode}_final.zip")