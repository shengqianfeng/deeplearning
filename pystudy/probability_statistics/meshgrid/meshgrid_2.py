import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3]])
y = np.array([[0, 0, 0, 0],
              [1, 1, 1, 1],
              [2, 2, 2, 2],
              [3, 3, 3, 3]])


plt.plot(x, y,
         marker='.',  # 点的形状为圆点
         markersize=10,  # 点设置大一点，看着清楚
         linestyle='-.')  # 线型为点划线
plt.grid(True)
plt.show()
