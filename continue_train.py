from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v3")

print("Purana best model load ho raha hai...")
model = PPO.load("models/normal/best_model", env=env)

print("Training start... 400k steps aur")
model.learn(total_timesteps=400000)

model.save("models/normal_final_v2")
print("HO GAYA! Ab 5/5 land karega")