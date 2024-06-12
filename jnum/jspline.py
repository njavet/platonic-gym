import numpy as np
from scipy.interpolate import CubicSpline as CS


class CubicSpline:
    def __init__(self, x_points, y_points):
        assert len(x_points) == len(y_points)
        self.n = len(x_points) - 1
        self.x_points = x_points
        self.y_points = y_points
        self.h = [self.x_points[i + 1] - self.x_points[i] for i in range(self.n)]
        self.c = np.zeros((4, self.n))
        self.compute_coefficients()

    def compute_cy_i(self, i):
        tmp0 = 3 * (self.y_points[i+1] - self.y_points[i]) / self.h[i]
        tmp1 = 3 * (self.y_points[i] - self.y_points[i-1]) / self.h[i-1]
        return tmp0 - tmp1

    def compute_b_i(self, i):
        tmp0 = (self.y_points[i + 1] - self.y_points[i]) / self.h[i]
        try:
            tmp1 = (self.h[i] / 3) * (self.c[1][i + 1] + 2 * self.c[1][i])
        except IndexError:
            tmp1 = (self.h[i] / 3) * 2 * self.c[1][i]
        return tmp0 - tmp1

    def compute_d_i(self, i):
        try:
            res = 1 / (3 * self.h[i]) * (self.c[1][i + 1] - self.c[1][i])
        except IndexError:
            res = 1 / (3 * self.h[i]) * (- self.c[1][i])
        return res

    def compute_coefficients(self):
        self.c[3] = self.y_points[:-1]

        mat = np.zeros((self.n - 1, self.n - 1))
        mat[0][0:2] = np.array([2*(self.h[0] + self.h[1]), self.h[1]])
        mat[-1][-3:] = np.array([self.h[-1], 2*(self.h[-1] + self.h[-2])])

        cx = np.zeros(self.n - 1)
        cx[0] = self.compute_cy_i(1)
        cx[1] = self.compute_cy_i(2)
        for i in range(2, self.n-1):
            mat[i, i-1] = self.h[i-1]
            mat[i, i] = 2 * (self.h[i-1] + self.h[i])
            mat[i, i+1] = self.h[i]
            cx[i] = self.compute_cy_i(i-1)
        self.c[1][1:] = np.linalg.solve(mat, cx)

        for i in range(self.n):
            self.c[2][i] = self.compute_b_i(i)
            self.c[0][i] = self.compute_d_i(i)


x = np.array([4, 6, 8, 10])
y = np.array([6, 3, 9, 0])
cs0 = CS(x, y, bc_type='natural')
print(cs0.c)

cs = CubicSpline(x, y)
print(cs.c)
