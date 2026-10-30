import PyFlyt.gym_envs

from clearml import Task
from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


task = Task.init(
    project_name="Drone-Thesis",
    task_name="PPO-waypoints-baseline-300x400",
    reuse_last_task_id=False,
)

cfg = task.connect({
    "seed": 0,
    "learning_rate": 1e-4,
    "timesteps": 800_000,
})

env = make_vec_env(
    "PyFlyt/QuadX-Waypoints-v4",
    n_envs=8,
    seed=cfg["seed"],
    wrapper_class=FlattenWaypointEnv,
    wrapper_kwargs={"context_length": 4},
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg["learning_rate"],
    policy_kwargs={"net_arch": [300, 400]},
    verbose=1,
    seed=cfg["seed"],
    tensorboard_log="./tb_logs/",
)

model.learn(
    total_timesteps=cfg["timesteps"],
    tb_log_name="PPO-waypoints-baseline_3.0",
)

model.save("ppo_waypoints_300x400")
env.close()
