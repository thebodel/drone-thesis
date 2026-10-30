import gymnasium
import PyFlyt.gym_envs

from stable_baselines3 import PPO
from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv


env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
env = FlattenWaypointEnv(env, context_length=4)
model = PPO.load("models/ppo_waypoints_300x400_continued_final_0.2")

obs, _ = env.reset()
term, trunc = False, False

while not (term or trunc):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, _ = env.step(action)

env.close()
