import numpy as np
import copy
class KalmanFilter:
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
    def setH(self, H):
        self.H = copy.copy(H)
    def setR(self, R):
        self.R = copy.copy(R)
    def setF(self, F):
        self.F = copy.copy(F)
    def setP(self, P):
        self.P = copy.copy(P)
    def setState(self, x):
        self.X = copy.copy(x)
    def setMeasurement(self, z):
        self.Z = copy.copy(z)
    def setAlpha(self, alpha):
        self.alpha = alpha

    def getQ(self):
        return copy.copy(self.Q)
    def getH(self):
        return copy.copy(self.H)
    def getR(self):
        return copy.copy(self.R)
    def getF(self):
        return copy.copy(self.F)
    def getState(self):
        return copy.copy(self.X)
    def getP(self):
        return copy.copy(self.P)


    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.transpose() + self.Q
    def update(self):
        self.K = self.P.dot(self.H.transpose()).dot(np.linalg.inv( self.H.dot(self.P).dot(self.H.transpose()) + self.R))
#         print("K_prime.shape:", K_prime.shape)
        self.X_ = self.X
        self.P_ = self.P
        self.X = self.X + self.K.dot(self.Z - self.H.dot(self.X))
        self.P = self.P - self.K.dot(self.H).dot(self.P)
    def warp(self, rotation, translation):
        self.X[:2, 0] = rotation @ self.X[:2, 0] + translation
        self.X[2:4, 0] = rotation @ self.X[2:4, 0]
        self.P[:2, :2] = rotation @ self.P[:2, :2] @ rotation.transpose()
        self.P[2:4, 2:4] = rotation @ self.P[2:4, 2:4] @ rotation.transpose()
    def adaptive_RQ(self):
        residual = (self.Z - self.H @ self.X).reshape(-1, 1)
        self.R = (1 - self.alpha) * residual @ residual.transpose() + self.H @ self.P_ @ self.H.transpose() + self.alpha * self.R
        innovation = (self.Z - self.H @ self.X_).reshape(-1, 1)
        self.Q = self.alpha * self.Q + (1 - self.alpha) * self.K @ innovation @ innovation.transpose() @ self.K.transpose()

def test_kalman_filter():
    kf = KalmanFilter()
    x = np.array([10.0, 10.0, 2.0, 2.0, 0.0, 0.0], dtype=float)
    kf.setState(x)
    z = np.array([10.5, 10.3, 1.4, 1.3, 0.0, 0.0], dtype=float)
    kf.setMeasurement(z)
    Q = np.identity(6)
    kf.setQ(Q)
    P = np.identity(6) * 0.5
    kf.setP(P)
    R = np.identity(6) * 0.3
    kf.setR(R)
    F = np.identity(6)
    delta_t = 0.2
    F[0, 2] = delta_t
    F[0, 4] = 0.5 * delta_t ** 2
    F[1, 3] = delta_t
    F[1, 5] = 0.5 * delta_t ** 2
    F[2, 4] = delta_t
    F[3, 5] = delta_t
    kf.setF(F)
    H = np.identity(6, dtype=float)
    kf.setH(H)
    kf.predict()
    kf.update()
    kf.setAlpha(0.7)
    kf.adaptive_RQ()
    X = kf.getState()
    P = kf.getP()
    print("X: ", X.transpose())
    print("P: ", P)

if __name__ == '__main__':
    test_kalman_filter()