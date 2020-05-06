import numpy as np
import matplotlib.pyplot as plt
import copy
class Trajectory:
    def __init__(self, time_duration, motion_pattern, hz):
        self.time_duration = time_duration
        # use polynomial to mimic a tracjectory
        # motion pattern should be a 2*N matrix, encoding x, y's equation to time t
        self._motion_pattern = np.array(motion_pattern)
        self._N = len(self._motion_pattern[0])
        self._hz = hz
        # self._trajectory (2* N)
        self._trajectory = self.generate_trajectory()
        # self._velocity (2* N)
        self._velocity = self.generate_velocity()
        self._timestamp = [float(i) / self._hz for i in range(self.time_duration * self._hz)]

    def generate_trajectory(self):
        t = [[(i / self._hz) ** n for n in reversed(range(self._N))]
                 for i in range(int(self.time_duration * self._hz))]
        t = np.array(t)
        traj = self._motion_pattern @ t.transpose()
        return traj

    def generate_velocity(self):
        v_pattern = [[(self._N - 1 - i) * self._motion_pattern[j][i]
                          for i in range(0, self._N - 1)]
                     for j in range(len(self._motion_pattern))]
        v_pattern = np.array(v_pattern)
        t = [[(i / self._hz) ** n for n in reversed(range(self._N - 1))]
                 for i in range(int(self.time_duration * self._hz))]
        t = np.array(t)
        velocity = v_pattern @ t.transpose()
        return velocity

    def get_frame_num(self):
        return self._N

    def get_velocity(self):
        return copy.copy(self._velocity)

    def get_position(self):
        return copy.copy(self._trajectory)
    def get_timestamps(self):
        return copy.copy(self._timestamp)

    def draw_yourself(self, explanation=""):
        plt.plot(self._trajectory[0], self._trajectory[1], marker='.', lw=2, label=explanation)
        plt.legend()
        plt.show()