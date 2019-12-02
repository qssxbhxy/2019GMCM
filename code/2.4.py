# 导入数据excel1
# 第一题最优路径
node = [0, 163, 114, 8, 309, 305, 123, 45, 160, 92, 93, 61, 292, 326]
shape = 327
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件2：数据集2-终稿.xlsx")
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)
fig = plt.figure()
ax = Axes3D(fig)
# 起点为黄色，终点为黄，中间点1为绿色+，0为黑色^
for i in range(shape):
    if i == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    elif 0 < i < 612 and correction_vector[i] == 1:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='g', marker='+')
    elif 0 < i < 612 and correction_vector[i] == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='b', marker='^')
    else:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    
step = 15
N = 100  # 画N个点做圆弧
# 构建node节点方向向量direction
direction = {}
x = [0, 0, 0]
y = [0, 0, 0]
#用差分算法生成圆弧
for i in range(len(node) - 1):
    direction[node[i], node[i + 1]] = [
        (coordinates_2[node[i + 1], 0] - coordinates_2[node[i], 0]) / A[node[i], node[i + 1]],
        (coordinates_2[node[i + 1], 1] - coordinates_2[node[i], 1]) / A[node[i], node[i + 1]],
        (coordinates_2[node[i + 1], 2] - coordinates_2[node[i], 2]) / A[node[i], node[i + 1]]]
for i in range(len(node) - 1):
    if i == 0:
        ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
                [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
                [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]], c='k', linewidth=1)
    else:
        x = [coordinates_2[node[i], 0], coordinates_2[node[i], 1], coordinates_2[node[i], 2]]
        y = [coordinates_2[node[i], 0], coordinates_2[node[i], 1], coordinates_2[node[i], 2]]
        for j in range(1, N):
            y[0] = y[0] + (j * direction[(node[i], node[i + 1])][0] + (N - j) * direction[(node[i - 1], node[i])][
                0]) / N * step
            y[1] = y[1] + (j * direction[(node[i], node[i + 1])][1] + (N - j) * direction[(node[i - 1], node[i])][
                1]) / N * step
            y[2] = y[2] + (j * direction[(node[i], node[i + 1])][2] + (N - j) * direction[(node[i - 1], node[i])][
                2]) / N * step
            ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], c='r', linewidth=2)
            x[0] = x[0] + (j * direction[(node[i], node[i + 1])][0] + (N - j) * direction[(node[i - 1], node[i])][
                0]) / N * step
            x[1] = x[1] + (j * direction[(node[i], node[i + 1])][1] + (N - j) * direction[(node[i - 1], node[i])][
                1]) / N * step
            x[2] = x[2] + (j * direction[(node[i], node[i + 1])][2] + (N - j) * direction[(node[i - 1], node[i])][
                2]) / N * step
        ax.plot([x[0], coordinates_2[node[i + 1], 0]], [x[1], coordinates_2[node[i + 1], 1]],
                [x[2], coordinates_2[node[i + 1], 2]], c='k')
