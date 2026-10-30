from pathlib import Path

import gymnasium
import PyFlyt.gym_envs

from clearml import Task
from numpy import random
from way_points.motor_failure import MotorFailure
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


PROJECT_DIR = Path(__file__).resolve().parent.parent


task = Task.init(
    project_name="Drone-Thesis",
    task_name="PPO-hover-continue",
    reuse_last_task_id=False,
)

cfg = task.connect({
    "seed": 0,
    "learning_rate": 7e-4,
    "extra_timesteps": 2_000_000,
    "failed_motor": [0, 3],
    "fail_after": [1, 299],
    "motor_efficiency": 0.2,
    "load_model": str(PROJECT_DIR / "ppo_hover_baseline_motor_failure_continued_0.2"),
    "save_model": str(PROJECT_DIR / "ppo_hover_baseline_motor_failure_continued_0.15"),
})


def make_train_env():
    env = gymnasium.make("PyFlyt/QuadX-Hover-v4")
    return MotorFailure(
        env,
        motor=cfg["failed_motor"],
        fail_after=cfg["fail_after"],
        efficiency=cfg["motor_efficiency"],
    )


env = make_vec_env(make_train_env, n_envs=32, seed=cfg["seed"])

model = PPO.load(
    cfg["load_model"],
    env=env,
    learning_rate=cfg["learning_rate"],
    policy_kwargs={"net_arch": [100, 120]},
    tensorboard_log=str(PROJECT_DIR / "tb_logs"),
)

model.learn(
    total_timesteps=cfg["extra_timesteps"],
    tb_log_name="PPO-hover-continue_motor_failure",
    reset_num_timesteps=False,
)

model.save(cfg["save_model"])
env.close()
