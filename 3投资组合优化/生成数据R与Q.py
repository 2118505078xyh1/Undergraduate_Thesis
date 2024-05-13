import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)
x5 = []
mean5 = []
x = np.random.beta(5, 10, size=1000) / 5-0.064
y = np.mean(x)
x5.append(list(x))
mean5.append(y)
x = np.random.beta(5, 5, size=1000) / 5-0.098
y = np.mean(x)
x5.append(list(x))
mean5.append(y)
x = np.random.beta(5, 1, size=1000) / 5-0.165
y = np.mean(x)
x5.append(list(x))
mean5.append(y)
x = np.random.beta(5, 10, size=1000) / 5-0.065
y = np.mean(x)
x5.append(list(x))
mean5.append(y)
x = np.random.beta(5, 15, size=1000) / 5-0.047
y = np.mean(x)
x5.append(list(x))
mean5.append(y)
cov55 = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        cov55[i][j] = np.cov(x5[i], x5[j])[0, 1]
print(mean5, cov55)
x5 = pd.DataFrame(x5)
x5.to_csv("x5.csv", index=False)
mean5 = pd.DataFrame(mean5)
mean5.to_csv("mean5.csv", index=False)
cov55 = pd.DataFrame(cov55)
cov55.to_csv("cov55.csv", index=False)

# 在数据集x上计算并画出直方图
# plt.hist(x, bins='auto', label=f'a={0.5}, b={0.6}')
# plt.legend()
# plt.show()
