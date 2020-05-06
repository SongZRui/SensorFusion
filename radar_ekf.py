import numpy as np
import math
import copy
class ExtendedKalmanFilter:
    def __init__(self):
        self.H = np.identity(6, dtype=float)
        self.R = np.identity(6, dtype=float)
        self.Q = np.identity(6, dtype=float)
        self.F = np.identity(6, dtype=float)
        self.X = np.zeros(6, dtype=float).reshape(6, 1)
        self.P = np.identity(6, dtype=float)
        self.Z = np.zeros(6, dtype=float).reshape(6, 1)
        self.alpha = 0.7

    def setQ(self, Q):
        self.Q = copy.copy(Q)
    def getQ(self):
        return copy.copy(self.Q)

    def setH(self, H):
        self.H = copy.copy(H)
    def getH(self):
        return copy.copy(self.H)

    def setR(self, R):
        self.R = copy.copy(R)
    def getR(self):
        return copy.copy(self.R)

    def setF(self, F):
        self.F = copy.copy(F)
    def getF(self):
        return copy.copy(self.F)

    def setP(self, P):
        self.P = copy.copy(P)
    def getP(self):
        return copy.copy(self.P)

    def setState(self, x):
        self.X = copy.copy(np.array(x, dtype=float).reshape(-1, 1))
    def getState(self):
        return copy.copy(self.X)

    def setMeasurement(self, z):
        self.Z = copy.copy(z)
    def setAlpha(self, alpha):
        self.alpha = alpha

    def predict(self):
        self.X = self.F.dot(self.X)
        self.P = self.F.dot(self.P).dot(self.F.transpose()) + self.Q
    def observe_wo_lat(self, state_):
        # defines nonlinear observation function
        state = copy.copy(state_)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        R = math.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        dRdt = vx * math.cos(theta) + vy * math.sin(theta)
        return np.array([R, theta, dRdt], dtype=float).reshape(-1, 1)

    def jacobian_wo_lat(self, state_):
        # param state is the linear expanding point, x, y, vx, vy
        state = copy.copy(state_)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        obs = self.observe_wo_lat(state)
        R, theta, dRdt = obs[0], obs[1], obs[2]
        dR_dx = x / R
        dR_dy = y / R
        dR_dvx = 0.0
        dR_dvy = 0.0
        dtheta_dx = - y / (R**2)
        dtheta_dy = x / (R**2)
        dtheta_dvx = 0.0
        dtheta_dvy = 0.0
        dRdt_dx = -vx * math.sin(theta) * dtheta_dx + vy * math.cos(theta) * dtheta_dx
        dRdt_dy = -vx * math.sin(theta) * dtheta_dy + vy * math.cos(theta) * dtheta_dy
        dRdt_dvx = math.sin(theta)
        dRdt_dvy = math.cos(theta)
        H = np.array([[dR_dx, dR_dy, dR_dvx, dR_dvy],
                      [dtheta_dx, dtheta_dy, dtheta_dvx, dtheta_dvy],
                      [dRdt_dx, dRdt_dy, dRdt_dvx, dRdt_dvy]], dtype=float)
        return H
    def observe_w_lat(self, state_):
        state = copy.copy(state_)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        R = math.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        dRdt = vx * math.cos(theta) + vy * math.sin(theta)
        dLdt = -vx * math.sin(theta) + vy * math.cos(theta)
        return np.array([R, theta, dRdt, dLdt], dtype=float).reshape(-1, 1)

    def jacobian_w_lat(self, state_):
        state = copy.copy(state_)
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        obs = self.observe_w_lat(state)
        R, theta, dRdt = obs[0], obs[1], obs[2]
        dR_dx = x / R
        dR_dy = y / R
        dR_dvx = 0.0
        dR_dvy = 0.0
        dtheta_dx = - y / (R**2)
        dtheta_dy = x / (R**2)
        dtheta_dvx = 0.0
        dtheta_dvy = 0.0
        dRdt_dx = -vx * math.sin(theta) * dtheta_dx + vy * math.cos(theta) * dtheta_dx
        dRdt_dy = -vx * math.sin(theta) * dtheta_dy + vy * math.cos(theta) * dtheta_dy
        dRdt_dvx = math.sin(theta)
        dRdt_dvy = math.cos(theta)
        dLdt_dx = -vx * math.cos(theta) * dtheta_dx - vy * math.sin(theta) * dtheta_dx
        dLdt_dy = -vx * math.cos(theta) * dtheta_dy - vy * math.sin(theta) * dtheta_dy
        dLdt_dvx = -math.sin(theta)
        dLdt_dvy = math.cos(theta)
        H = np.array([[dR_dx, dR_dy, dR_dvx, dR_dvy],
                      [dtheta_dx, dtheta_dy, dtheta_dvx, dtheta_dvy],
                      [dRdt_dx, dRdt_dy, dRdt_dvx, dRdt_dvy],
                      [dLdt_dx, dLdt_dy, dLdt_dvx, dLdt_dvy]], dtype=float)
        return H

    def update_wo_lat(self):
        # update jacobian matrix
        self.H = self.jacobian_wo_lat(self.X)
        self.K = self.P.dot(self.H.transpose()).dot(np.linalg.inv( self.H.dot(self.P).dot(self.H.transpose()) + self.R))
        self.X_ = copy.copy(self.X)
        self.P_ = copy.copy(self.P)
        self.X = self.X + np.array(self.K.dot(self.Z - self.observe_wo_lat(self.X)))
        self.P = self.P - self.K.dot(self.H).dot(self.P)

    def update_w_lat(self):
        # update jacobian matrix
        self.H = self.jacobian_w_lat(self.X)
        self.K = self.P.dot(self.H.transpose()).dot(np.linalg.inv( self.H.dot(self.P).dot(self.H.transpose()) + self.R))
        self.X_ = copy.copy(self.X)
        self.P_ = copy.copy(self.P)
        self.X = self.X + np.array(self.K.dot(self.Z - self.observe_w_lat(self.X)))
        self.P = self.P - self.K.dot(self.H).dot(self.P)

    def warp(self, rotation, translation):
        self.X[:2, 0] = rotation @ self.X[:2, 0] + translation
        self.X[2:4, 0] = rotation @ self.X[2:4, 0]
        self.P[:2, :2] = rotation @ self.P[:2, :2] @ rotation.transpose()
        self.P[2:4, 2:4] = rotation @ self.P[2:4, 2:4] @ rotation.transpose()
    def adaptive_RQ_wo_lat(self):
        residual = (self.Z - self.observe_wo_lat(self.X)).reshape(-1, 1)
        self.R = (1 - self.alpha) * residual @ residual.transpose() + self.H @ self.P_ @ self.H.transpose() + self.alpha * self.R
        innovation = (self.Z - self.observe_wo_lat(self.X_)).reshape(-1, 1)
        self.Q = self.alpha * self.Q + (1 - self.alpha) * self.K @ innovation @ innovation.transpose() @ self.K.transpose()

    def adaptive_RQ_w_lat(self):
        residual = (self.Z - self.observe_w_lat(self.X)).reshape(-1, 1)
        self.R = (1 - self.alpha) * residual @ residual.transpose() + self.H @ self.P_ @ self.H.transpose() + self.alpha * self.R
        innovation = (self.Z - self.observe_w_lat(self.X_)).reshape(-1, 1)
        self.Q = self.alpha * self.Q + (1 - self.alpha) * self.K @ innovation @ innovation.transpose() @ self.K.transpose()

def test_ekf():
    ekf = ExtendedKalmanFilter()
    vx = 5
    x0 = np.array([5, 10.0, 4.8, 0.0], dtype=float).reshape(-1, 1)
    ekf.setState(x0)
    sqrt_2 = math.sqrt(2.0)
    delta_t = 1
    # after 1 second, x1 = [10, 10, 5, 0]
    theta = math.atan2(10, 10)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    rv = vx * cos_theta
    lv = -vx * sin_theta

    # z = np.array([10 * sqrt_2 + 0.5, np.pi / 4, rv, lv], dtype=float).reshape(-1, 1)
    z = np.array([10 * sqrt_2 + 0.5, np.pi / 4, rv], dtype=float).reshape(-1, 1)
    ekf.setMeasurement(z)
    Q = np.identity(4) * 0.001
    ekf.setQ(Q)
    P = np.identity(4) * 0.005
    P[2, 2] *= 20
    ekf.setP(P)
    R = np.identity(3) * 0.003
    R[0, 0] = 0.5
    ekf.setR(R)
    F = np.identity(4)
    F[0, 2] = delta_t
    F[1, 3] = delta_t
    ekf.setF(F)
    ekf.predict()
    X = ekf.getState()
    print('after predict: ', X.transpose())
    ekf.update_wo_lat()
    X = ekf.getState()
    print("X: ", X.transpose())
#     ekf.setAlpha(0.7)
#     ekf.adaptive_RQ_wo_lat()
    P = ekf.getP()

    print("P: ", P)
test_ekf()