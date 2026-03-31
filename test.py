import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class ProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0

            ep = len(self.episode_rewards)
            avg = np.mean(self.episode_rewards[-10:])
            print(f"Episode {ep:>4} | Reward: {self.episode_rewards[-1]:>6.1f} | Avg (last 10): {avg:>6.1f}")

        return True

env = gym.make("CartPole-v1", render_mode="human")
model = PPO("MlpPolicy", env, verbose=0)  # verbose=0 since we handle printing ourselves

model.learn(total_timesteps=50_000, callback=ProgressCallback())

env.close()
