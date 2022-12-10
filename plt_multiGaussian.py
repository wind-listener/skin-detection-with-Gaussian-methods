import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv, det


# 多维高斯分布
def gaussion(x, mu, Sigma):
    dim = len(x)
    constant = (2 * np.pi) ** (-dim / 2) * det(Sigma) ** (-0.5)
    return constant * np.exp(-0.5 * (x - mu).dot(inv(Sigma)).dot(x - mu))


# 混合高斯模型
def gaussion_mixture(x, Pi, mu, Sigma):
    z = 0
    for idx in range(len(Pi)):
        z += Pi[idx] * gaussion(x, mu[idx], Sigma[idx])
    return z


def pltMG(mu, cov, alpha):
    x = np.linspace(-1, 1, 150)
    y = np.linspace(-1, 1, 150)
    x, y = np.meshgrid(x, y)
    X = np.array([x.ravel(), y.ravel()]).T
    z = [gaussion_mixture(x, alpha, mu, cov) for x in X]
    z = np.array(z).reshape(x.shape)
    fig = plt.figure()
    # 绘制3d图形
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x, y, z)
    # 绘制等高线
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(x, y, z)
    plt.show()


# mu = [[0.3421303, 0.44560197], [0.00170464, 0.00454446]]
# cov = [[[1.57215564e-02, 1.90945083e-02], [1.90945083e-02, 2.46057727e-02]],
#        [[8.36192996e-06, 8.33263252e-06], [8.33263252e-06, 3.80993367e-05]]]
# alpha = [0.86968153, 0.13031847]
# pltMG(mu, cov, alpha)