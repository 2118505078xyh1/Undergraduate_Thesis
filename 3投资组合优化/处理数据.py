import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

cov = pd.read_csv("cov55.csv", ).values
mean1 = pd.read_csv("mean5.csv", ).values.transpose()
mean2 = mean1[0]
Q = cov
# print(Q, mean2)
W = np.array([[0.1, 0.2, 0.1, 0.6, 0.3]]).transpose()
A = np.array([mean2, [1, 1, 1, 1, 1]])
list_rp = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, ]
std5 = []
fx5 = []
for rp in list_rp:
    b = np.array([[rp, 1]]).transpose()
    h1 = np.hstack((Q, -A.transpose()))
    h2 = np.hstack((A, np.zeros((2, 2))))
    H = np.vstack((h1, h2))
    H_inv = np.linalg.inv(H)
    B = np.vstack((np.zeros((5, 1)), b))
    WW = np.dot(H_inv, B)
    w = WW[0:5]
    print(w.shape)
    if w[4] < 0:
        w = w / (1 - w[4])
        w[4] = 0
    for i in range(5):
        w[i][0] = round(w[i][0], 3)
    std0 = np.std(w)
    std5.append(std0)
    fx = np.dot(np.dot(w.transpose(), Q), w)[0][0] * 10000  # 方差风险
    fx5.append(fx)
    for i in range(5):
        w[i][0] = round(w[i][0] * 100, 2)
    print(rp, std0, fx,
          str(w[0][0]) + ',' + str(w[1][0]) + ',' + str(w[2][0]) + ',' + str(w[3][0]) + ',' + str(w[4][0]))
print(std5, list_rp, fx5)

plt.plot(list_rp, std5)
plt.xlabel('value of rp')
plt.ylabel('Standard Deviation')
plt.grid(True)
plt.show()

plt.plot(list_rp, fx5)
plt.xlabel('value of rp')
plt.ylabel('Variance Risk')
plt.grid(True)
plt.show()
plt.plot(fx5, std5)
plt.xlabel('Variance Risk')
plt.ylabel('Standard Deviation')
plt.grid(True)
plt.show()
