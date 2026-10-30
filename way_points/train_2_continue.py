import PyFlyt.gym_envs

import gymnasium
from clearml import Task
from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from way_points.motor_failure import MotorFailure



class WaypointCompletionEvalCallback(BaseCallback):
    def __init__(
        self,
        task,
        eval_freq=20_000,
        eval_episodes=20,
        failed_motor=0,
        fail_after=100,
        motor_efficiency=0.0,
        goal_reach_distance=0.2,
    ):
        super().__init__()
        self.task = task
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.last_eval_timestep = -eval_freq
        eval_env = gymnasium.make(
            "PyFlyt/QuadX-Waypoints-v4",
            goal_reach_distance=goal_reach_distance,
        )
        eval_env = MotorFailure(
            eval_env,
            motor=failed_motor,
            fail_after=fail_after,
            efficiency=motor_efficiency,
        )
        self.eval_env = FlattenWaypointEnv(eval_env, context_length=4)

    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        targets_reached = []
        env_complete = []

        for _ in range(self.eval_episodes):
            obs, info = self.eval_env.reset()
            term, trunc = False, False

            while not (term or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self.eval_env.step(action)

            targets_reached.append(info.get("num_targets_reached", 0))
            env_complete.append(float(info.get("env_complete", False)))

        logger = self.task.get_logger()
        logger.report_scalar(
            "eval_waypoints",
            "targets_reached_mean",
            sum(targets_reached) / len(targets_reached),
            iteration=self.num_timesteps,
        )
        logger.report_scalar(
            "eval_waypoints",
            "env_complete_rate",
            sum(env_complete) / len(env_complete),
            iteration=self.num_timesteps,
        )
        self.task.flush(wait_for_uploads=False)

        return True

    def _on_training_end(self):
        self.eval_env.close()


task = Task.init(
    project_name="Drone-Thesis",
    task_name="PPO-waypoints-motor-failure-eff-0.35",
    reuse_last_task_id=False,
)

cfg = task.connect({
    "seed": 0,
    "learning_rate": 8e-5,
    "extra_timesteps": 1_000_000,
    "eval_freq": 80_000,
    "eval_episodes": 1000,
    "failed_motor": [0, 3],
    "fail_after": [30, 250],
    "motor_efficiency": 0.2,
    "goal_reach_distance": 0.4,
    "load_model": "models/ppo_waypoints_motor_failure_eff_0.3_20_35",
    "save_model": "models/ppo_waypoints_motor_failure_eff_0.3_20_30",
})


def make_train_env():
    env = gymnasium.make(
        "PyFlyt/QuadX-Waypoints-v4",
        goal_reach_distance=cfg["goal_reach_distance"],
    )
    env = MotorFailure(
        env,
        motor=cfg["failed_motor"],
        fail_after=cfg["fail_after"],
        efficiency=cfg["motor_efficiency"],
    )
    return FlattenWaypointEnv(env, context_length=4)


env = make_vec_env(make_train_env, n_envs=12, seed=cfg["seed"])

model = PPO.load(
    cfg["load_model"],
    env=env,
    learning_rate=cfg["learning_rate"],
    tensorboard_log="./tb_logs/",
)

completion_eval_callback = WaypointCompletionEvalCallback(
    task,
    eval_freq=cfg["eval_freq"],
    eval_episodes=cfg["eval_episodes"],
    failed_motor=cfg["failed_motor"],
    fail_after=cfg["fail_after"],
    motor_efficiency=cfg["motor_efficiency"],
    goal_reach_distance=cfg["goal_reach_distance"],
)

model.learn(
    total_timesteps=cfg["extra_timesteps"],
    tb_log_name="PPO-waypoints-motor-failure-eff-0.3",
    reset_num_timesteps=False,
    callback=completion_eval_callback,
)

model.save(cfg["save_model"])
env.close()
