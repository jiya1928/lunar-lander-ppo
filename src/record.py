import gymnasium as gym
from stable_baselines3 import PPO
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
model = PPO.load("models/normal_final_v2")

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, f"./videos/{mode}/", episode_trigger=lambda x: True)

for i in range(5):
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total += reward
        if truncated:
            break
    print(f"Video Episode {i+1}: Score {total:.2f}")

env.close()
print(f"Videos saved in videos/{mode}/")