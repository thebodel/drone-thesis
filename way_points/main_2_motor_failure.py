import gymnasium
import PyFlyt.gym_envs

from stable_baselines3 import PPO
from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv
from way_points.motor_failure import MotorFailure

env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
#env = MotorFailure(env, motor=1, fail_after=70, efficiency=0.2)
env = FlattenWaypointEnv(env, context_length=4)
model = PPO.load("models/ppo_waypoints_motor_failure_eff_0.3_20_30")

obs, _ = env.reset()
term, trunc = False, False

while not (term or trunc):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, _ = env.step(action)

env.close()
