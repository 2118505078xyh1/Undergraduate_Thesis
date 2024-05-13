import numpy as np
from matplotlib import pyplot as plt
import time

A = np.random.normal(0.0, 1.0, (2000, 3000))
x_ture = np.zeros((3000, 1))
for _ in range(300):
    x_ture[_ * 10] = 1
    x_ture[_ * 10 + 5] = -1
b = np.dot(A, x_ture)
x = np.ones((3000, 1))
lambda0 = np.zeros((2000, 1))
beta = 1
v = 0.9
mu = 0.5
k = 0
p = []
t1 = time.time()  # begin time
while True:
    x1 = np.max(np.hstack((x + beta * np.dot(A.transpose(), lambda0) - beta, np.zeros((3000, 1)))), axis=1) + np.min(
        np.hstack((x + beta * np.dot(A.transpose(), lambda0) + beta, np.zeros((3000, 1)))), axis=1)
    x1 = x1.reshape(3000, 1)
    lambda1 = lambda0 - beta * (np.dot(A, x) - b)
    r = beta * np.linalg.norm(
        np.vstack((np.dot(A.transpose(), lambda0 - lambda1), np.dot(A, x - x1)))) / np.linalg.norm(
        np.vstack((x - x1, lambda0 - lambda1)))
    while r > v:
        beta = (2 / 3) * beta * min(1, 1 / r)
        x1 = np.max(np.hstack((x + beta * np.dot(A.transpose(), lambda0) - beta, np.zeros((3000, 1)))),
                    axis=1) + np.min(
            np.hstack((x + beta * np.dot(A.transpose(), lambda0) + beta, np.zeros((3000, 1)))), axis=1)
        x1 = x1.reshape(3000, 1)
        lambda1 = lambda0 - beta * (np.dot(A, x) - b)
        r = beta * np.linalg.norm(
            np.vstack((np.dot(A.transpose(), lambda0 - lambda1), np.dot(A, x - x1)))) / np.linalg.norm(
            np.vstack((x - x1, lambda0 - lambda1)))
    x2 = np.max(np.hstack((x + beta * np.dot(A.transpose() , lambda1)- beta, np.zeros((3000, 1)))), axis=1) + \
         np.min(np.hstack((x + beta * np.dot(A.transpose() , lambda1)+ beta, np.zeros((3000, 1)))), axis=1)
    x2 = x2.reshape(3000, 1)
    lambda0 = lambda0 - beta * (np.dot(A, x1) - b)
    if r <= mu:
        beta = 1.5 * beta
    p.append(np.linalg.norm(x2 - x))
    print(beta, r, k, np.linalg.norm(x2 - x), )
    if k==500 or k==1000:
        t2 = time.time()  # end time
        t = t2 - t1  # compute running time
        print(t)
    if np.linalg.norm(x2 - x) < 0.0001 or k > 200:
        break
    k = k + 1
    x = x2
t2 = time.time()  # end time
t = t2 - t1  # compute running time
print(t,k,'000')
plt.plot(p)
# 添加标题和轴标签
plt.xlabel('number of iterations')
plt.ylabel('||xk-xk-1||2')
# 显示网格
plt.grid(True)
# 显示图像
plt.show()
