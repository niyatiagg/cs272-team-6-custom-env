import gymnasium as gym
import highway_env
import time
from custom_env import AccidentEnv

def pretty_print_rewards(info):
    """Pretty-print the reward breakdown for better readability."""
    rewards = info.get('rewards', {})
    if not rewards:
        print("  No rewards info found.")
        return
    print("\n  Reward Breakdown:")
    for k, v in rewards.items():
        marker = " <==" if v < 0 else ""
        print(f"    {k:22}: {v: .3f}{marker}")
    print("  " + "-"*32)

# Registering the environment
gym.register("accident-v0", entry_point="custom_env:AccidentEnv")

env = gym.make(
    "accident-v0", 
    render_mode="human", 
    config={
        "manual_control": True,
        "observation": {"type": "LidarObservation"},
        "duration": 20
    }
)

obs, info = env.reset()
terminated = truncated = False

while not terminated and not truncated:
    time.sleep(0.2)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Improved, clear printout:
    print(f"\nAction: {info.get('action')} | Speed: {info.get('speed', 0): .2f} | Crashed: {info.get('crashed')}")
    print(f"Step Reward: {reward: .3f}")
    pretty_print_rewards(info)

env.close()
