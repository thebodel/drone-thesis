import gymnasium
import PyFlyt.gym_envs
import matplotlib.pyplot as plt

from way_points.motor_failure import MotorFailure
from stable_baselines3 import PPO


env = gymnasium.make("PyFlyt/QuadX-Hover-v4", render_mode="human",)
env = MotorFailure(env, motor=[0, 3], fail_after=100, efficiency=0.2)
model = PPO.load("ppo_hover_baseline_motor_failure_continued_0.2")

obs, _ = env.reset()
term, trunc = False, False
positions = []

while not (term or trunc):
    positions.append(env.unwrapped.env.state(0)[3].copy())
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, _ = env.step(action)

env.close()

xs = [position[0] for position in positions]
ys = [position[1] for position in positions]
zs = [position[2] for position in positions]

plt.figure()
plt.plot(xs, ys)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Center of mass trajectory: XY")
plt.grid(True)
plt.axis("equal")

plt.figure()
plt.plot(zs)
plt.xlabel("step")
plt.ylabel("z")
plt.title("Center of mass height")
plt.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Center of mass trajectory: 3D")

plt.show()
