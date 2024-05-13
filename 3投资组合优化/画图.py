import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = [i for i in range(0, 1001)]
x5 = pd.read_csv("x5.csv", ).values
ly1 = [1]
ly2 = [1]
ly3 = [1]
ly4 = [1]
ly5 = [1]
for i in range(1000):
    y1 = ly1[-1]*(x5[0][i]+1)
    y2 = ly2[-1]*(x5[1][i]+1)
    y3 = ly3[-1]*(x5[2][i]+1)
    y4 = ly4[-1]*(x5[3][i]+1)
    y5 = ly5[-1]*(x5[4][i]+1)
    ly1.append(y1)
    ly2.append(y2)
    ly3.append(y3)
    ly4.append(y4)
    ly5.append(y5)
# 创建一个图形窗口
plt.figure()
# 在同一窗口中绘制多条曲线
plt.plot(x, ly1, label='beta(5,10)')
plt.plot(x, ly2, label='beta(5,5)')
plt.plot(x, ly3, label='beta(5,1)')
plt.plot(x, ly4, label='beta(5,10)')
plt.plot(x, ly5, label='beta(5,15)')
# 添加图例
plt.legend()
# 添加标题和轴标签
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
