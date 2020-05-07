#!/usr/bin/env python3

from trajectory import Trajectory
from lidar_observation import LidarObservation

import random
import copy
import numpy as np
class LiDARObservationGenerator:
    def __init__(self, ego, obj, position_sigma=0.1, velocity_sigma=0.3, velocity_mu=-1.0, seed=1000):
        self._ego = copy.copy(ego)
        self._obj = copy.copy(obj)
        # self._position_obs (2 * N)
        # self._velocity_obs (2 * N)
        random.seed(seed) # for repeatable results
        self._position_obs, self._velocity_obs = self.generate_local_obs()
        self._gt_lidar_obss = self.generate_lidar_obs()
        self.add_noise(self._position_obs, sigma=position_sigma, mu=0.0)
        self.add_noise(self._velocity_obs, mu=velocity_mu, sigma=velocity_sigma)
        self._noisy_lidar_obss = self.generate_lidar_obs()

        self._obj_position_world = self.local2world(self._position_obs, is_obs_vector=False)
        self._obj_velocity_world = self.local2world(self._velocity_obs, is_obs_vector=True)
        self._timestamp = self._ego.get_timestamps()

    def get_lidar_obs(self):
        return copy.copy(self._noisy_lidar_obss)

    def get_lidar_gt(self):
        return copy.copy(self._gt_lidar_obss)

    def generate_lidar_obs(self):
        lidar_obss = []
        for i in range(len(self._position_obs[0])):
            lidar_obs = LidarObservation(self._position_obs[0][i], self._position_obs[1][i],
                                         self._velocity_obs[0][i], self._velocity_obs[1][i],
                                         self._poses[i])
            lidar_obss.append(lidar_obs)
        return lidar_obss

    def calculate_ori_in_world_coordinate(self, traj):
        velocity = traj.get_velocity()
        orientations = [ np.arctan2(velocity[1][i], velocity[0][i]) for i in range(len(velocity[0]))]
        positions = traj.get_position()

        poses = [[[np.cos(orientations[i]), -np.sin(orientations[i]), positions[0][i]],
                  [np.sin(orientations[i]), np.cos(orientations[i]), positions[1][i]],
                  [0.0, 0.0, 1.0]] for i in range(len(orientations))]
        poses = np.asarray(poses)
        oris = [[[np.cos(orientations[i]), -np.sin(orientations[i])],
                    [np.sin(orientations[i]), np.cos(orientations[i])]] for i in range(len(orientations))]
        oris = np.array(oris)
        return oris, poses

    def generate_local_obs(self):
        # return obj position and absolute velocity in ego car coodinate
        # ego car coordinate is defined as followed:
        #    x axis points to the front of ego car
        #    y axis points to the right of ego car
        #    assume ego car orientation is always same as velocity direction
        self.ego_oris, self._poses = self.calculate_ori_in_world_coordinate(self._ego)
#         print("ego_oris: ", ego_oris)
        diff_position_world_coor = self._obj.get_position() - self._ego.get_position()
        print("diff_position_world_coor.shape:", diff_position_world_coor.shape)
        diff_position_ego_coor = np.array([self.ego_oris[i].transpose() @ diff_position_world_coor[:, i]
                                               for i in range(len(self.ego_oris))])
        obj_velocity_world_coor = self._obj.get_velocity()
        obj_velocity_ego_coor = np.array([self.ego_oris[i].transpose() @ obj_velocity_world_coor[:, i]
                                              for i in range(len(self.ego_oris))])
        return diff_position_ego_coor.transpose(), obj_velocity_ego_coor.transpose()

    def add_noise(self, obs, noise_option="const_variance", sigma = 0.3, mu = 0.04):
        if noise_option == "const_variance":
            for i in range(len(obs[0])):
                obs[0][i] += random.gauss(mu, sigma)
                obs[1][i] += random.gauss(mu, sigma)
        elif noise_option == "variant_variance":
            for i in range(len(obs[0])):
                scale = random.random()*2
                sigma = scale * sigma
                obs[0][i] += random.gauss(mu, sigma)
                obs[1][i] += random.gauss(mu, sigma)
        elif noise_option == "linear_variance":
            scale = 1.2
            N = len(obs[0])
            for i in range(len(obs[0])):
                sigma_tmp = scale * sigma * float(i) / N
                obs[0][i] += random.gauss(mu, sigma_tmp)
                obs[1][i] += random.gauss(mu, sigma_tmp)
        else:
            print("%s not implemented yes"%noise_option)

    def local2world(self, obs, is_obs_vector):
        if is_obs_vector==True:
            print("TRUE obs.shape:", obs.shape)
            obs_world = [self.ego_oris[i] @ obs[:, i] for i in range(len(self.ego_oris))]

        else:
            ego_positions = self._ego.get_position()
            print("ego_positions.shape:", ego_positions.shape)
            print("self.ego_oris.shape:", self.ego_oris.shape)
            print("obs.shape: ", obs.shape)
            obs_world = [self.ego_oris[i] @ obs[:, i] + ego_positions[:, i] for i in range(len(self.ego_oris))]

        return np.array(obs_world).transpose()

    def get_warp(self, from_index, to_index):
#         print("from_pose:", self._poses[from_index])
#         print("to_pose:", self._poses[to_index])
        warp = np.linalg.inv(self._poses[to_index]) @ self._poses[from_index]
        return warp

    def get_timestamps(self):
        return copy.copy(self._timestamp)
    def draw_yourself(self, explanation=""):
        import matplotlib.pyplot as plt
        fig = LidarObservation.draw_lidar_obss(self._noisy_lidar_obss, show=False, label="Noisy")
        fig = LidarObservation.draw_lidar_obss(self._gt_lidar_obss, fig=fig, show=True, label="GT")


def test_lidar():
    obj_motion_pattern = [[0, 0, 3, 0], [0, 0, 0, 0]]
    obj_traj = Trajectory(3, obj_motion_pattern, 10)
    ego_motion_pattern = [[0, 0, 0, 5], [0, 0, 1, 5]]
    ego_traj = Trajectory(3, ego_motion_pattern, 10)
#     obj_traj.draw_yourself(explanation="obj")
#     ego_traj.draw_yourself(explanation="ego")
    lidar_traj = LiDARObservationGenerator(ego_traj, obj_traj)
    lidar_traj.draw_yourself()
if __name__=='__main__':
    test_lidar()