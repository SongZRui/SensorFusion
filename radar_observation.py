#!/usr/bin/env python3
from lidar_observation import LidarObservation
import copy
import numpy as np


class RadarObservation:
    def __init__(self, R, theta, r_v, l_v, Twr, r_a=0.0):
        self._R = copy.copy(R)
        self._theta = copy.copy(theta)
        self._r_v = copy.copy(r_v)
        self._l_v = copy.copy(l_v)
        self._Twr = np.array(copy.copy(Twr))
        self._Rwr = self._Twr[:2, :2]
        self._twr = self._Twr[:2, 2]

        ct = np.cos(self._theta)
        st = np.sin(self._theta)
        self._local_x = self._R * ct
        self._local_y = self._R * st
        self._local_vx = self._r_v * ct - self._l_v * st
        self._local_vy = self._r_v * st + self._l_v * ct
        local = np.array([self._local_x, self._local_y])
        global_ = self._Rwr @ local + self._twr
        self._global_x = global_[0]
        self._global_y = global_[1]
        local = np.array([self._local_vx, self._local_vy])
        global_ = self._Rwr @ local
        self._global_vx = global_[0]
        self._global_vy = global_[0]

    def get_radar_measurement(self):
        return np.array([self._R, self._theta, self._r_v, self._l_v])

    def get_lidar_format_measurement(self):
        return LidarObservation(self._local_x, self._local_y, self._local_vx, self._local_vy, self._Twr)

    @staticmethod
    def draw_radar_obss(obss, show=True, fig=None, label=""):
        '''obss should be a seriase of radar observation on same object'''
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=(16, 18))
        thetas = [obs._theta for obs in obss]
        index = [i for i in range(len(thetas))]
        ax = fig.add_subplot(3, 4, 1)
        ax.set_title("radar theta(rad)")
        ax.set_ylabel("rad")
        ax.plot(index, thetas, marker='.')

        Rs = [obs._R for obs in obss]
        ax = fig.add_subplot(3, 4, 2)
        ax.set_title("radar range(m)")
        ax.set_ylabel("m")
        ax.plot(index, Rs, marker='.')

        r_vs = [obs._r_v for obs in obss]
        ax = fig.add_subplot(3, 4, 3)
        ax.set_title("radar range velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, r_vs, marker='.')

        l_vs = [obs._l_v for obs in obss]
        ax = fig.add_subplot(3, 4, 4)
        ax.set_title("radar lateral velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, l_vs, marker='.')

        local_xs = [obs._local_x for obs in obss]
        ax = fig.add_subplot(3, 4, 5)
        ax.set_title("radar local x(m)")
        ax.set_ylabel("m")
        ax.plot(index, local_xs, marker='.')

        local_ys = [obs._local_y for obs in obss]
        ax = fig.add_subplot(3, 4, 6)
        ax.set_title("radar local y(m)")
        ax.set_ylabel("m")
        ax.plot(index, local_ys, marker='.')

        local_vxs = [obs._local_vx for obs in obss]
        ax = fig.add_subplot(3, 4, 7)
        ax.set_title("radar local x velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, local_vxs, marker='.')

        local_vys = [obs._local_vy for obs in obss]
        ax = fig.add_subplot(3, 4, 8)
        ax.set_title("radar local y velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, local_vys, marker='.')

        global_xs = [obs._global_x for obs in obss]
        ax = fig.add_subplot(3, 4, 9)
        ax.set_title("radar global x(m)")
        ax.set_ylabel("m")
        ax.plot(index, global_xs, marker='.')

        global_ys = [obs._global_y for obs in obss]
        ax = fig.add_subplot(3, 4, 10)
        ax.set_title("radar global y(m)")
        ax.set_ylabel("m")
        ax.plot(index, global_ys, marker='.')

        global_vxs = [obs._global_vx for obs in obss]
        ax = fig.add_subplot(3, 4, 11)
        ax.set_title("radar global x velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, global_vxs, marker='.')

        global_vys = [obs._global_vy for obs in obss]
        ax = fig.add_subplot(3, 4, 12)
        ax.set_title("radar global y velocity(m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, global_vys, marker='.', label=label)
        plt.legend()
        if show == True:
            plt.show()
        return fig


def test_radar_observation():
    Twr = np.identity(3)
    Twr[0, 2] = 10

    radar1 = RadarObservation(10, 0.3, 3, 2, Twr=Twr)
    radars = [radar1]
    RadarObservation.draw_radar_obss(radars)


if __name__ == "__main__":
    test_radar_observation()
