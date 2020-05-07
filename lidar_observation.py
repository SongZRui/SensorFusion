#!/usr/bin/env python3

import copy
import numpy as np


class LidarObservation:
    def __init__(self, local_px, local_py, local_vx, local_vy, Twl, local_ax=0.0, local_ay=0.0):
        self._local_px = copy.copy(local_px)
        self._local_py = copy.copy(local_py)
        self._local_vx = copy.copy(local_vx)
        self._local_vy = copy.copy(local_vy)
        self._local_ax = copy.copy(local_ax)
        self._local_ay = copy.copy(local_ay)

        self._Twl = np.array(copy.copy(Twl))
        self._Rwl = self._Twl[:2, :2]
        self._twl = self._Twl[:2, 2]

        local = np.array([self._local_px, self._local_py])
        global_ = self._Rwl @ local + self._twl
        self._global_px = global_[0]
        self._global_py = global_[1]
        local = np.array([self._local_vx, self._local_vy])
        global_ = self._Rwl @ local
        self._global_vx = global_[0]
        self._global_vy = global_[0]

    def get_local_px(self):
        return copy.copy(self._local_px)

    def get_local_py(self):
        return copy.copy(self._local_py)

    def get_global_px(self):
        return copy.copy(self._global_px)

    def get_global_py(self):
        return copy.copy(self._global_py)

    @staticmethod
    def draw_lidar_obss(obss, show=True, fig=None, label=""):
        '''obss should be a seriase of lidar observation on same object'''
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=(16, 18))
        local_pxs = [obs._local_px for obs in obss]
        index = [i for i in range(len(local_pxs))]
        ax = fig.add_subplot(3, 4, 1)
        ax.set_title("lidar px[LOCAL](m)")
        ax.set_ylabel("m")
        ax.plot(index, local_pxs, marker='.')

        local_pys = [obs._local_py for obs in obss]
        ax = fig.add_subplot(3, 4, 2)
        ax.set_title("lidar py[LOCAL](m)")
        ax.set_ylabel("m")
        ax.plot(index, local_pys, marker='.')

        local_vxs = [obs._local_vx for obs in obss]
        ax = fig.add_subplot(3, 4, 3)
        ax.set_title("lidar vx[LOCAL](m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, local_vxs, marker='.')

        local_vys = [obs._local_vy for obs in obss]
        ax = fig.add_subplot(3, 4, 4)
        ax.set_title("lidar vy[LOCAL](m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, local_vys, marker='.')

        theta = np.arctan2(local_vys, local_vxs)
        ax = fig.add_subplot(3, 4, 5)
        ax.set_title("lidar velocity orientation (rad)")
        ax.set_ylabel("rad")
        ax.plot(index, theta, marker='.')

        global_pxs = [obs._global_px for obs in obss]
        print("global_pxs:", global_pxs)
        ax = fig.add_subplot(3, 4, 6)
        ax.set_title("lidar px[GLOBAL](m)")
        ax.set_ylabel("m")
        ax.plot(index, global_pxs, marker='.')

        global_pys = [obs._global_py for obs in obss]
        ax = fig.add_subplot(3, 4, 7)
        ax.set_title("lidar py[GLOBAL](m)")
        ax.set_ylabel("m")
        ax.plot(index, global_pys, marker='.')

        global_vxs = [obs._global_vx for obs in obss]
        ax = fig.add_subplot(3, 4, 8)
        ax.set_title("lidar vx[GLOBAL](m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, global_vxs, marker='.')

        global_vys = [obs._global_vy for obs in obss]
        ax = fig.add_subplot(3, 4, 9)
        ax.set_title("lidar vy[GLOBAL](m/s)")
        ax.set_ylabel("m/s")
        ax.plot(index, global_vys, marker='.')

        theta = np.arctan2(global_vys, global_vxs)
        ax = fig.add_subplot(3, 4, 10)
        ax.set_title("lidar velocity orientation[GLOBAL] (rad)")
        ax.set_ylabel("rad")
        ax.plot(index, theta, marker='.')

        ax = fig.add_subplot(3, 4, 11)
        ax.set_title("lidar local trajectory")
        ax.set_ylabel("y")
        ax.plot(local_pxs, local_pys, marker='.')

        local_vys = [obs._local_vy for obs in obss]
        ax = fig.add_subplot(3, 4, 12)
        ax.set_title("lidar global trajectory")
        ax.set_ylabel("y")
        ax.plot(global_pxs, global_pys, marker='.', label=label)
        plt.legend()
        if show == True:
            plt.show()
        return fig


def test_lidar_observation():
    Twl = np.identity(3)
    Twl[0, 2] = 10

    lidar1 = LidarObservation(10, 0.3, 3, 2, Twl=Twl)
    lidar2 = LidarObservation(13, 0.8, 3.8, 2.6, Twl=Twl)
    lidars = [lidar1, lidar2]
    LidarObservation.draw_lidar_obss(lidars)


if __name__ == "__main__":
    test_lidar_observation()
