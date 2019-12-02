
node = [0, 169, 322, 270, 89, 236, 132, 53, 112, 268, 250, 243, 73, 249, 274, 12, 216, 16, 282, 141, 291, 161, 326]
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件2：数据集2-终稿.xlsx")
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[326] = 3
fig = plt.figure()
ax = Axes3D(fig)
# 起点为红色，终点为绿色，中间点1为黄色+，0为黄色^
for i in range(327):
    if i == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    elif 0 < i < 326 and correction_vector[i] == 1:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='r', marker='+')
    elif 0 < i < 326 and correction_vector[i] == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='b', marker='^')
    else:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
for i in range(len(node) - 1):
    ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
            [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
            [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]], c='k', linewidth=1)
