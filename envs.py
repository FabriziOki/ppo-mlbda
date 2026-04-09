import gymnasium as gym
import minigrid
envs = [e for e in gym.envs.registry.keys() if "MiniGrid" in e]
print("\n".join(map(str, envs)))
