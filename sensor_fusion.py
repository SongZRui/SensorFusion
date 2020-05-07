#!/usr/bin/env python3

from lidar_observation import LidarObservation
from radar_observation import RadarObservation
from trajectory import Trajectory
from lidar_observation_generator import LiDARObservationGenerator
from radar_observation_generator import RadarObservationGenerator

from kalman_filter import KalmanFilter

import numpy as np
import copy
import matplotlib.pyplot as plt


class SensorFusion:
    X_INDEX = 0
    Y_INDEX = 1
    VX_INDEX = 2
    VY_INDEX = 3
    AX_INDEX = 4
    AY_INDEX = 5

    def __init__(self):
        self.resetup([True] * 6)
        self._kf = KalmanFilter()

    def resetup(self, maintain_state_flag):
        self._maintain_state_flag = copy.copy(maintain_state_flag)
        self._maintain_state_size = self._maintain_state_flag.count(True)
        print(self._maintain_state_size)
        print(self._maintain_state_flag)
        # lidar H will not change even if we change the state we maintain
        self._lidar_H = np.zeros(self._maintain_state_size * 2).reshape(2, -1)
        self._lidar_H[0, 0], self._lidar_H[1, 1] = 1.0, 1.0
    '''setup F matrix, if we maintain acceleration,
        it should be reflected in F matrix'''

    def setup_F(self, dt):
        self._F = np.identity(self._maintain_state_size)
        self._F[SensorFusion.X_INDEX, SensorFusion.VX_INDEX] = dt
        self._F[SensorFusion.Y_INDEX, SensorFusion.VY_INDEX] = dt

        if self._maintain_state_flag[SensorFusion.AX_INDEX]:
            self._F[SensorFusion.VX_INDEX, SensorFusion.AX_INDEX] = dt
        if self._maintain_state_flag[SensorFusion.AY_INDEX]:
            self._F[SensorFusion.VY_INDEX, SensorFusion.AY_INDEX] = dt

    def setup_lidar_H(self):
        self._kf.setH(self._lidar_H)

    def update_with_lidar_obs(self, lidar_obs):
        z = np.array([lidar_obs.get_local_px(),
                      lidar_obs.get_local_py()]).reshape(-1, 1)
        self._kf.setMeasurement(z)
        self._kf.setH(self._lidar_H)
        print("lidar_H ", self._lidar_H)
        self._kf.update()

        return self._kf.getState(), self._kf.getP()

    def predict(self, dt, warp):
        rotation = warp[:2, :2]
        translation = warp[:2, 2]
        self._kf.warp(rotation, translation)
        self.setup_F(dt)
        self._kf.setF(self._F)
        self._kf.predict()

    def update_with_radar_obs(self, radar_obs, use_lat_v=False):
        z = radar_obs.get_radar_measurement().reshape(-1, 1)
        if not use_lat_v:
            z = z[:3, :].reshape(-1, 1)
        print("radar z", z)
        self._kf.setMeasurement(z)
        self.setup_radar_H(use_lat_v)
        print("radar H", self._kf.getH())
        self._kf.update()
        return self._kf.getState(), self._kf.getP()

    def setup_radar_H(self, use_lat_v=False):
        if use_lat_v:
            self.setup_radar_H_w_lat(self.get_state())
        else:
            self.setup_radar_H_wo_lat(self.get_state())

    def cal_radar_observe_wo_lat(self, state):
        state = copy.copy(state)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        R = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        dRdt = vx * np.cos(theta) + vy * np.sin(theta)
        return np.array([R, theta, dRdt]).reshape(-1, 1)

    def setup_radar_H_wo_lat(self, state):
        state = copy.copy(state)
        state.reshape(-1, 1)
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        obs = self.cal_radar_observe_wo_lat(state)
        R, theta, dRdt = obs[0, 0], obs[1, 0], obs[2, 0]
        dR_dx = x / R
        dR_dy = y / R
        dR_dvx = 0.0
        dR_dvy = 0.0
        dtheta_dx = - y / (R**2)
        dtheta_dy = x / (R**2)
        dtheta_dvx = 0.0
        dtheta_dvy = 0.0
        dRdt_dx = -vx * np.sin(theta) * dtheta_dx + \
            vy * np.cos(theta) * dtheta_dx
        dRdt_dy = -vx * np.sin(theta) * dtheta_dy + \
            vy * np.cos(theta) * dtheta_dy
        dRdt_dvx = np.sin(theta)
        dRdt_dvy = np.cos(theta)
        H = np.array([[dR_dx, dR_dy, dR_dvx, dR_dvy],
                      [dtheta_dx, dtheta_dy, dtheta_dvx, dtheta_dvy],
                      [dRdt_dx, dRdt_dy, dRdt_dvx, dRdt_dvy]])
        self._kf.setH(H)

    def cal_radar_observe_w_lat(self, state_):
        state = copy.copy(state_)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        R = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        dRdt = vx * np.cos(theta) + vy * np.sin(theta)
        dLdt = -vx * np.sin(theta) + vy * np.cos(theta)
        return np.array([R, theta, dRdt, dLdt], dtype=float).reshape(-1, 1)

    def setup_radar_H_w_lat(self, state):
        state = copy.copy(state)
        state.reshape(-1, 1)
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        obs = self.cal_radar_observe_w_lat(state)
        R, theta, dRdt = obs[0, 0], obs[1, 0], obs[2, 0]
        dR_dx = x / R
        dR_dy = y / R
        dR_dvx = 0.0
        dR_dvy = 0.0
        dtheta_dx = - y / (R**2)
        dtheta_dy = x / (R**2)
        dtheta_dvx = 0.0
        dtheta_dvy = 0.0
        dRdt_dx = -vx * np.sin(theta) * dtheta_dx + \
            vy * np.cos(theta) * dtheta_dx
        dRdt_dy = -vx * np.sin(theta) * dtheta_dy + \
            vy * np.cos(theta) * dtheta_dy
        dRdt_dvx = np.sin(theta)
        dRdt_dvy = np.cos(theta)
        dLdt_dx = -vx * np.cos(theta) * dtheta_dx - \
            vy * np.sin(theta) * dtheta_dx
        dLdt_dy = -vx * np.cos(theta) * dtheta_dy - \
            vy * np.sin(theta) * dtheta_dy
        dLdt_dvx = -np.sin(theta)
        dLdt_dvy = np.cos(theta)
        H = np.array([[dR_dx, dR_dy, dR_dvx, dR_dvy],
                      [dtheta_dx, dtheta_dy, dtheta_dvx, dtheta_dvy],
                      [dRdt_dx, dRdt_dy, dRdt_dvx, dRdt_dvy],
                      [dLdt_dx, dLdt_dy, dLdt_dvx, dLdt_dvy]])
        self._kf.setH(H)

    def set_maintain_state(self, state_flags):
        self.resetup(state_flags)

    def set_P(self, P):
        self._kf.setP(P[:self._maintain_state_size,
                        :self._maintain_state_size])

    def get_P(self):
        return self._kf.getP()

    def set_Q(self, Q):
        self._kf.setQ(Q[:self._maintain_state_size,
                        :self._maintain_state_size])

    def get_Q(self):
        return self._kf.getQ()

    def set_R(self, R):
        self._kf.setR(R)

    def get_R(self):
        return self._kf.getR()

    def set_state(self, state):
        print("state:", state)
        if len(state) != self._maintain_state_size:
            print(" STATE dimension not equals to maintain state size")
        s = []
        for i in range(len(self._maintain_state_flag)):
            if self._maintain_state_flag[i]:
                s.append(state[i])
        s = np.array(s).reshape(-1, 1)
        self._kf.setState(s)

    def get_state(self):
        return self._kf.getState()

    def get_state_in_lidar_format(self, Twl):
        state = self.get_state()
        px, py, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        if len(state) == 6:
            ax, ay = state[4, 0], state[5, 0]
        else:
            ax, ay = 0, 0
        return LidarObservation(px, py, vx, vy, Twl=Twl, local_ax=ax, local_ay=ay)

    def get_state_in_radar_format(self, Twr):
        state = self.get_state()
        px, py, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        R = np.sqrt(px ** 2 + py ** 2)
        theta = np.arctan2(py, px)
        range_v = vx * np.cos(theta) + vy * np.sin(theta)
        lat_v = -vx * np.sin(theta) + vy * np.cos(theta)
        return RadarObservation(R, theta, range_v, lat_v, Twr)


def test_sensor_fusion_with_lidar():
    motion_pattern = [[0, 0, 2, 3], [0, 0, 2, 1]]
    time_duration = 3
    frequence = 10
    ego_traj = Trajectory(time_duration, motion_pattern, frequence)
    motion_pattern = [[0, 0, 2, 0], [0, 0, 0, 0]]
    obj_traj = Trajectory(time_duration, motion_pattern, frequence)

    lidar_obs_generator = LiDARObservationGenerator(
        ego_traj, obj_traj, position_sigma=0.01, velocity_sigma=0.1, velocity_mu=0.0)
    lidar_measurement = lidar_obs_generator.get_lidar_obs()
    lidar_gt = lidar_obs_generator.get_lidar_gt()
    timestamps = lidar_obs_generator.get_timestamps()

    fusion = SensorFusion()
    fusion.resetup([True, True, True, True, True, True])
    lidar_R = np.identity(2) * 0.001
    fusion.set_R(lidar_R)
    Q = np.identity(6) * 0.1
    fusion.set_Q(Q)
    P = np.identity(6)
    fusion.set_P(P)
    initial_state = [lidar_measurement[0]._local_px,
                     lidar_measurement[0]._local_py]
    initial_state.append(lidar_measurement[0]._local_vx)
    initial_state.append(lidar_measurement[0]._local_vy)
    initial_state.append(lidar_measurement[0]._local_ax)
    initial_state.append(lidar_measurement[0]._local_ay)

    initial_state = np.array(initial_state).reshape(-1, 1)
    fusion.set_state(initial_state)
    fusion_results = []
    for i in range(1, len(lidar_measurement)):
        dt = timestamps[i] - timestamps[i-1]
        warp = ego_traj.get_warp(i-1, i)
        fusion.predict(dt, warp)
        fusion.update_with_lidar_obs(lidar_measurement[i])
        fusion_results.append(
            fusion.get_state_in_lidar_format(lidar_measurement[i]._Twl))

    fig = LidarObservation.draw_lidar_obss(
        fusion_results, label='fusion_result', show=False)
    fig = LidarObservation.draw_lidar_obss(
        lidar_gt, label="GT", fig=fig, show=False)
    LidarObservation.draw_lidar_obss(
        lidar_measurement, label="measurement", fig=fig, show=True)


def test_sensor_fusion_with_radar():
    motion_pattern = [[0, 0, 0.1, -3], [0, 0, 0, 0]]
    time_duration = 3
    frequence = 10
    ego_traj = Trajectory(time_duration, motion_pattern, frequence)
    motion_pattern = [[0, 0, 0, 0], [0, 0, 0, 0]]
    obj_traj = Trajectory(time_duration, motion_pattern, frequence)

    lidar_obs_generator = LiDARObservationGenerator(
        ego_traj, obj_traj, position_sigma=0.01, velocity_sigma=0.1, velocity_mu=0.0)
    lidar_gt = lidar_obs_generator.get_lidar_gt()
    timestamps = lidar_obs_generator.get_timestamps()

    radar_obs_generator = RadarObservationGenerator(
        ego_traj, obj_traj, R_sigma=0.01)
    radar_measurement = radar_obs_generator.get_radar_measurements()
    radar_gt = radar_obs_generator.get_radar_gt()

    fusion = SensorFusion()
    fusion.resetup([True, True, True, True, False, False])
    lidar_R = np.identity(4) * 0.001
    fusion.set_R(lidar_R)
    Q = np.identity(4) * 0.01
    fusion.set_Q(Q)
    P = np.identity(4) * 0.01
    fusion.set_P(P)
    initial_state = [lidar_gt[0]._local_px,
                     lidar_gt[0]._local_py]
    initial_state.append(lidar_gt[0]._local_vx)
    initial_state.append(lidar_gt[0]._local_vy)
    initial_state.append(lidar_gt[0]._local_ax)
    initial_state.append(lidar_gt[0]._local_ay)
    initial_state = np.array(initial_state).reshape(-1, 1)
    fusion.set_state(initial_state)

    fusion_results = []
    radar_measurements_lidar_format = []
    fusion_results_radar_format = []
    for i in range(1, len(radar_measurement)):
        dt = timestamps[i] - timestamps[i-1]
        warp = ego_traj.get_warp(i-1, i)
        fusion.predict(dt, warp)
        fusion.update_with_radar_obs(radar_measurement[i], use_lat_v=True)
        fusion_results.append(
            fusion.get_state_in_lidar_format(lidar_gt[i]._Twl))
        radar_measurements_lidar_format.append(
            radar_measurement[i].get_lidar_format_measurement())
        fusion_results_radar_format.append(
            fusion.get_state_in_radar_format(radar_measurement[i]._Twr))

    fig = LidarObservation.draw_lidar_obss(
        fusion_results, label='fusion_result', show=False)
    fig = LidarObservation.draw_lidar_obss(
        lidar_gt, label="GT", show=False, fig=fig)
    LidarObservation.draw_lidar_obss(
        radar_measurements_lidar_format, label='radar_z', show=False, fig=fig)

    fig2 = RadarObservation.draw_radar_obss(
        fusion_results_radar_format, show=False, label="fusion_result")
    fig2 = RadarObservation.draw_radar_obss(
        radar_gt, show=False, fig=fig2, label='radar_gt')
    RadarObservation.draw_radar_obss(
        radar_measurement, show=False, fig=fig2, label='radar_measurement')
    plt.show()


if __name__ == '__main__':
    # test_sensor_fusion_with_lidar()
    test_sensor_fusion_with_radar()
