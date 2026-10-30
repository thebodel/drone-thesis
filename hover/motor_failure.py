import gymnasium


class MotorFailure(gymnasium.Wrapper):
    def __init__(self, env, motor=0, fail_after=100, efficiency=0.0):
        super().__init__(env)
        self.motor = motor
        self.fail_after = fail_after
        self.efficiency = efficiency

    def _sample_motor(self):
        if isinstance(self.motor, (list, tuple)):
            low, high = self.motor
            return int(self.unwrapped.np_random.integers(low, high + 1))
        return int(self.motor)

    def _sample_fail_after(self):
        if isinstance(self.fail_after, (list, tuple)):
            low, high = self.fail_after
            return int(self.unwrapped.np_random.integers(low, high + 1))
        return int(self.fail_after)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.current_motor = self._sample_motor()
        self.current_fail_after = self._sample_fail_after()
        self.motors = self.unwrapped.env.drones[0].motors
        return obs, info

    def step(self, action):
        if self.steps == self.current_fail_after:
            self.motors.thrust_coef[self.current_motor] *= self.efficiency
            self.motors.torque_coef[self.current_motor] *= self.efficiency

        self.steps += 1
        return self.env.step(action)
