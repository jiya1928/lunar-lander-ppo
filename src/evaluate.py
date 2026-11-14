import gymnasium as gym
from stable_baselines3 import PPO
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
n_episodes = 20

model = PPO.load(f"models/{mode}_final")
env = gym.make("LunarLander-v3")

scores = []
for i in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total += reward
        if truncated:
            break
    scores.append(total)
    print(f"Episode {i+1}: {total:.2f}")

avg = sum(scores) / len(scores)
print(f"\n=== {mode.upper()} ===")
print(f"Average over {n_episodes}: {avg:.2f}")
print(f"Solved? {avg >= 200} (200+ = Solved)")
env.close()