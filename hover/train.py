import gymnasium
import PyFlyt.gym_envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from clearml import Task
from way_points.motor_failure import MotorFailure


task = Task.init(
    project_name="Drone-Thesis",
    task_name="PPO-hover-baseline",
    reuse_last_task_id=False,
)

cfg = task.connect({
    "seed": 0,
    "learning_rate": 3e-4,
    "timesteps": 2_000_000,
    "failed_motor": 0,
    "fail_after": 100,
    "motor_efficiency": 0.0,
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



model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg["learning_rate"],
    policy_kwargs={"net_arch": [200, 140]},
    verbose=1,
    seed=cfg["seed"],
    tensorboard_log="./tb_logs/",
)
model.learn(
    total_timesteps=cfg["timesteps"],
    tb_log_name="PPO-hover-baseline_motor_failure",
)

model.save("ppo_hover_baseline_motor_failure")
env.close()
