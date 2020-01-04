import matplotlib.pyplot as plt
# TODO:梯度下降，代码实现
XOld = 0
XNew = 6
# 误差
eps = 0.00002
# 步长
alpha = 0.01
# 原函数
def f(x):
    return x ** 4 - 3 * x ** 3 + 2
# 导函数
def f_prime(x):
    return 4 * x ** 3 - 9 * x ** 2
x = [];
y=[];
if __name__ == "__main__":
    while abs(f(XOld) - f(XNew)) > eps:
        XOld = XNew
        XNew = XOld - alpha * f_prime(XOld)
        x.append(XNew)
        y.append(f(XNew))
        print(XNew, f(XNew))
# print("Final value is:", XNew, f(XNew))
plt.plot([-5, 5], [0,0], '--g')
plt.plot([0,0], [-5, 5], '--r')
# o 实心圈标记
plt.plot(x, y, 'o')
plt.grid()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()