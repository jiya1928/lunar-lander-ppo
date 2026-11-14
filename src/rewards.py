import gymnasium as gym

# 1. NORMAL - Kuch change nahi, built-in reward hi
class NormalReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

# 2. TIME PENALTY - Har frame pe thoda charge, taaki jaldi land kare
class TimePenaltyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = reward - 0.05  # har frame -0.05, late karne pe nuksaan
        return obs, reward, done, truncated, info

# 3. HACKED - Ye isko bigad dega, hover karna sikhega
class HackedReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # obs[1]=height, obs[3]=vertical speed, obs[5]=tilt
        # Agar seedha hai aur slow hai, to +1 bonus
        if abs(obs[4]) < 0.1 and abs(obs[3]) < 0.2 and obs[1] > 0.2:
            reward = reward + 1.0
        return obs, reward, done, truncated, info

# 4. FIXED - Hack ka fix, sirf improvement pe reward + time penalty
class FixedReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_shaping = None
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Built-in reward already shaping hai, hum bas time penalty add karte hai
        # Aur hover ka bonus nahi dete
        reward = reward - 0.05
        return obs, reward, done, truncated, info