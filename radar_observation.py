import random
import numpy as np
import matplotlib.pyplot as plt
import copy.copy


class RadarTrajectory:
    def __init__(self, ego, obj, R_sigma=0.1, range_v_sigma=0.01, theta_sigma=0.1 / 180.0 * np.pi, lat_v_sigma=0.1):
        random.seed(1001) # for repeatable result
        self._ego = copy.copy(ego)
        self._obj = copy.copy(obj)
        self._R = self.generate_R()
        self._theta, self._range_v, self._lat_v = self.generate_theta_range_v()
        self.add_noise(self._R, sigma = R_sigma)
        self.add_noise(self._range_v, sigma=range_v_sigma)
        self.add_noise(self._lat_v, sigma=lat_v_sigma)
        self.add_noise(self._theta, sigma = theta_sigma)
        self._obj_position_world, self._obj_velocity_world = self.obs2world(R=self._R, theta=self._theta,
                                                                            range_v=self._range_v, lat_v=self._lat_v)
#         self._range_v = np.asarray(range_velocity)
        self._timestamp = self._ego.get_timestamps()

    def generate_R(self):
        # generate range observation, and this is only relative to position difference
        R = self._ego.get_position() - self._obj.get_position()
        R = np.linalg.norm(R, axis=0)
        return R

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

    def generate_theta_range_v(self):
        # return obj position and absolute velocity in ego car coodinate
        # ego car coordinate is defined as followed:
        #    x axis points to the front of ego car
        #    y axis points to the right of ego car
        #    assume ego car orientation is always same as velocity direction
        # ego_oris: N * 2 * 2, where N is number of frames
        self.ego_oris, self.poses = self.calculate_ori_in_world_coordinate(self._ego)
        # diff_position_in_world: N * 2
        diff_position_in_world = (self._obj.get_position() - self._ego.get_position()).transpose()
        # diff_position_in_ego: N * 2
        diff_position_in_ego = np.array([self.ego_oris[i].transpose() @ diff_position_in_world[i].transpose()
                                             for i in range(len(diff_position_in_world))])

        observing_angles = [np.arctan2(diff_position_in_ego[i][1], diff_position_in_ego[i][0])
                                for i in range(len(diff_position_in_ego))]
        observing_angles = np.array(observing_angles)

        obj_velocity_world_coor = self._obj.get_velocity()
        obj_velocity_ego_coor = np.array([self.ego_oris[i].transpose() @ obj_velocity_world_coor[:, i]
                                              for i in range(len(obj_velocity_world_coor[0]))])

        range_v = [np.cos(observing_angles[i]) * obj_velocity_ego_coor[i][0] + np.sin(observing_angles[i]) * obj_velocity_ego_coor[i][1]
                        for i in range(len(obj_velocity_ego_coor))]
        range_v = np.array(range_v)

        lat_v = [np.cos(observing_angles[i]) * obj_velocity_ego_coor[i][1] - np.sin(observing_angles[i]) * obj_velocity_ego_coor[i][0]
                        for i in range(len(obj_velocity_ego_coor))]
        lat_v = np.asarray(lat_v)
        return observing_angles, range_v, lat_v

    def add_noise(self, obs, noise_option="const_variance", sigma = 0.3, mu = 0.04):
        import random
        if noise_option == "const_variance":
            for i in range(len(obs)):
                obs[i] += random.gauss(mu, sigma)
                obs[i] += random.gauss(mu, sigma)
        elif noise_option == "variant_variance":
            for i in range(len(obs)):
                scale = random.random() * 2
                sigma = scale * sigma
                obs[i] += random.gauss(mu, sigma)
                obs[i] += random.gauss(mu, sigma)
        elif noise_option == "linear_variance":
            scale = 1.2
            N = len(obs)
            for i in range(len(obs)):
                sigma_tmp = scale * sigma * float(i) / N
                obs[i] += random.gauss(mu, sigma_tmp)
                obs[i] += random.gauss(mu, sigma_tmp)
        else:
            print("%s not implemented yes"%noise_option)

    def obs2world(self, theta, R, range_v, lat_v=0.0):
        xy_local = [[np.cos(theta[i]) * R[i], np.sin(theta[i]) * R[i]] for i in range(len(R))]
        xy_local = np.array(xy_local).transpose() # in shape (2 * N)
        ego_xy_world = self._ego.get_position()
        xy_world = [self.ego_oris[i] @ xy_local[:, i] + ego_xy_world[:, i] for i in range(len(self.ego_oris))]
        xy_world = np.array(xy_world).transpose()
        vxvy_local = [[-np.sin(theta[i]) * lat_v[i] + np.cos(theta[i]) * range_v[i],
                       np.sin(theta[i]) * range_v[i] + np.cos(theta[i]) * lat_v[i]]
                     for i in range(len(theta))]
        vxvy_local = np.array(vxvy_local).transpose()
        vxvy_world = [self.ego_oris[i] @ vxvy_local[:, i] for i in range(len(self.ego_oris))]
        vxvy_world = np.array(vxvy_world).transpose()
        return xy_world, vxvy_world

    def get_thetas(self):
        return self._theta
    def get_theta(self,  i):
        return self._theta[i]
    def get_Rs(self):
        return self._R
    def get_R(self, i):
        return self._R[i]
    def get_range_vs(self):
        return self._range_v
    def get_range_v(self, i):
        return self._range_v[i]
    def get_lat_v(self, i):
        return self._lat_v[i]
    def get_timestamps(self):
        return copy.copy(self._timestamp)
    def draw_yourself(self, explanation=""):
        x = np.linspace(0, len(self._R), len(self._R))
        plt.plot(x, self._R, label = explanation + '-range')
        plt.legend()
        plt.show()
        plt.plot(x, np.rad2deg(self._theta), label = explanation + '-theta')
        plt.legend()
        plt.show()
        plt.plot(x, self._range_v, label = explanation + '-range_v')
        plt.legend()
        plt.show()
        plt.plot(x, self._lat_v, label = explanation + '-lat_v')
        plt.legend()
        plt.show()
        plt.plot(self._obj_position_world[0], self._obj_position_world[1], label=explanation+'-xy_world')
        plt.legend()
        plt.show()
        plt.plot(x, self._obj_velocity_world[0], label=explanation+'-vx_world')
        plt.plot(x, self._obj_velocity_world[1], label=explanation+'-vy_world')
        plt.legend()
        plt.show()
def test_radar():
    obj_motion_pattern = [[0, 0, 3, 0], [0, 0, 0, 0]]
    obj_traj = Trajectory(3, obj_motion_pattern, 10)
    obj_traj.draw_yourself(explanation="obj_trajectory")

    ego_motion_pattern = [[0, 0, 1, 5], [0, 0, 1, 5]]
    ego_traj = Trajectory(3, ego_motion_pattern, 10)
    ego_traj.draw_yourself(explanation="ego_trajectory")
    radar_traj = RadarTrajectory(ego_traj, obj_traj)
    radar_traj.draw_yourself()
# test_radar()