import matplotlib.pyplot as plt
import numpy as np


x_d3 = np.linspace(0.01, 0.20, 20)
y_d3 = [0.0028, 0.0075, 0.0134, 0.0188, 0.0288, 0.0371, 0.0499, 0.0669, 0.0817, 0.0977, 0.1227, 0.1384, 0.1567, 0.1809, 0.2112, 0.2275, 0.2608, 0.2831, 0.2989, 0.3211]

plt.figure(dpi=600)
plt.plot(x_d3, y_d3, label='sur_d3')

x_d5 = [0.01, 0.11]
y_d5 = [0.0014, 0.088]
plt.plot(x_d5, y_d5, 'x', markersize=6, color='red', label='sur_d5')  # 使用'x'来表示叉，蓝色，大小为10

x_d7 = [0.01]
y_d7 = [0.0021]
plt.plot(x_d7, y_d7, 'x', markersize=6, color='green', label='sur_d7')  # 使用'x'来表示叉，蓝色，大小为10

plt.xlabel('Physical error rate')
plt.ylabel('Logical error rate')
plt.legend()
plt.show()
plt.savefig('../img/logical_error_rate.png')
