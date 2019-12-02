# 论文代码
## 简要说明 
    求解主要采用python语言，调用了规划求解器Gurobi的接口。另外，罚函数系数程序中可以自行调试，本文档中系数仅做参考。
--------------------------------------------------------------
## 问题一：Excel1数据集的求解
~~~python
import numpy as np
import pandas as pd
from gurobipy import *

# 0.读取excel表2的数据
excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点； 第二列为x坐标，第三列为y坐标，第四列为z坐标
coordinates_2 = excel2.values[1:, 1:4]  
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[612] = 3
# 计算欧式距离，用矩阵A标记
A = np.zeros((613, 613))
for i in range(613):
    for j in range(613):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)
# 导入第一题的基本常量
shape = 613
alpha1 = 25
alpha2 = 15
beta1 = 20
beta2 = 25
delta = 0.001
theta = 30
# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量

# 2.建立模型
model1 = Model()
# 添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x = model1.addVars(shape, shape, vtype=GRB.BINARY, name='x')
# 添加新变量h[i],v[i],i=0 to shape-1
h = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h")
v = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v")
# 添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
# 添加限制条件起始点，中间点，终点的出度入度条件
test1 = [0] * shape  # 出度表达式
test2 = [0] * shape  # 入度表达式
# test1[i]表示i节点的出度
for (i, j) in edge:
    test1[i] = test1[i] + x[i, j]
# test2[i]表示i节点的入度
for (j, i) in edge:
    test2[i] = test2[i] + x[j, i]
for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)
# 添加限制条件对h，v进行约束
for (i, j) in edge:
    model1.addConstr(correction_vector[i] * h[i] + delta * A[i, j] - h[j] <= 10000 - 10000 * x[i, j])
    model1.addConstr((1 - correction_vector[i]) * v[i] + delta * A[i, j] - v[j] <= 10000 - 10000 * x[i, j])
for i in range(shape):
    if i == 0:
        model1.addConstr(h[i] == 0)
        model1.addConstr(v[i] == 0)
    elif 0 < i < shape - 1:
        if correction_vector[i] == 1:
            model1.addConstr(h[i] <= alpha2)
            model1.addConstr(v[i] <= alpha1)
        else:
            model1.addConstr(h[i] <= beta2)
            model1.addConstr(v[i] <= beta1)
    else:
        model1.addConstr(h[i] <= theta)
        model1.addConstr(v[i] <= theta)
# 添加目标函数
# 设置罚函数系数
dict_modify = {}
for i in range(shape):
    for j in range(shape):
        dict_modify[i, j] = 10000
dict_modify = tupledict(dict_modify)
model1.setObjective(dict_A.prod(x) + 1 * dict_modify.prod(x), GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if s.x == 1:
        print('%s''%g' % (s.varName, s.x))
print('Obj:%g' % model1.objVal)
~~~
## 问题一：绘制最优航迹图的程序
~~~python
# 导入数据excel1
# 第一题最优路径
node = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

excel1 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx")
coordinates_2 = excel1.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel1.values[1:, 4])
correction_vector[0] = 2
correction_vector[612] = 3
fig = plt.figure()
ax = Axes3D(fig)
# 起点，终点为黄色，中间点1为绿+，0为蓝^
for i in range(613):
    if i == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    elif 0 < i < 612 and correction_vector[i] == 1:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='g', marker='+')
    elif 0 < i < 612 and correction_vector[i] == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='b', marker='^')
    else:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
ax.scatter(list(coordinates_2[1:613, 0]), list(coordinates_2[1:613:, 1]), list(coordinates_2[1:613:, 2]), c='y',
           marker='.')
#起点终点涂色
ax.scatter(0, 50000, 5000, c='r')
ax.scatter(100000, 59652.3433795158, 5022.00116448164, c='g')
for i in range(len(node) - 1):
    ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
            [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
            [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]], c='k', linewidth=1)
~~~

## 问题一：Excel2数据集的求解
~~~python
import numpy as np
import pandas as pd
from gurobipy import *

# 导入第一题的基本常量
shape = 327
alpha1 = 20
alpha2 = 10
beta1 = 15
beta2 = 20
delta = 0.001
theta = 20
# 0.读取excel表2的数据
excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件2：数据集2-终稿.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
# 计算欧式距离，用矩阵A标记
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)
# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量

# 2.建立模型
model1 = Model()
# 添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x = model1.addVars(shape, shape, vtype=GRB.BINARY, name='x')
# 添加新变量h[i],v[i],i=0 to shape-1
h = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h")
v = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v")
# 添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
# 添加限制条件起始点，中间点，终点的出度入度条件
test1 = [0] * shape  # 出度表达式
test2 = [0] * shape  # 入度表达式
# test1[i]表示i节点的出度
for (i, j) in edge:
    test1[i] = test1[i] + x[i, j]
# test2[i]表示i节点的入度
for (j, i) in edge:
    test2[i] = test2[i] + x[j, i]
for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)
# 添加限制条件对h，v进行约束
for (i, j) in edge:
    model1.addConstr(correction_vector[i] * h[i] + delta * A[i, j] - h[j] <= 10000 - 10000 * x[i, j])
    model1.addConstr((1 - correction_vector[i]) * v[i] + delta * A[i, j] - v[j] <= 10000 - 10000 * x[i, j])
for i in range(shape):
    if i == 0:
        model1.addConstr(h[i] == 0)
        model1.addConstr(v[i] == 0)
    elif 0 < i < shape - 1:
        if correction_vector[i] == 1:
            model1.addConstr(h[i] <= alpha2)
            model1.addConstr(v[i] <= alpha1)
        else:
            model1.addConstr(h[i] <= beta2)
            model1.addConstr(v[i] <= beta1)
    else:
        model1.addConstr(h[i] <= theta)
        model1.addConstr(v[i] <= theta)
    # 添加目标函数
    # 添加罚函数系数
dict_modify = {}
for i in range(shape):
    for j in range(shape):
        dict_modify[i, j] = 10000
dict_modify = tupledict(dict_modify)
model1.setObjective(dict_A.prod(x) + 0 * dict_modify.prod(x), GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if s.x == 1:
        print('%s''%g' % (s.varName, s.x))
print('Obj:%g' % model1.objVal)
~~~
## 问题一excel2：绘制最优航迹图的程序
~~~python
# 导入数据excel2
# 第一题最优路径
node = [0, 163, 114, 8, 309, 305, 123, 45, 160, 92, 93, 61, 292, 326]
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
    # ax.scatter(list(coordinates_2[1:327,0]),list(coordinates_2[1:327:,1]),list(coordinates_2	[1:327:,2]),c='y',marker='.',markersize=3)
# ax.scatter(0,50000,5000,c='r')
# ax.scatter(100000,74860.55,5499.61,c='g')
# ax.plot([coordinates_2[0,0],coordinates_2[326,0]],
# [coordinates_2[0,1],coordinates_2[326,1]],
# [coordinates_2[0,2],coordinates_2[326,2]],c='k',linewidth=1)

for i in range(len(node) - 1):
    ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
            [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
            [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]], c='k', linewidth=1)
~~~
## 问题二：Excel1数据集的求解
~~~python
import copy

import numpy as np
import pandas as pd
from gurobipy import *

# 导入基本常量
# shape=613
# alpha1=25
# alpha2=15
# beta1=20
# beta2=25
# delta=0.001
# theta=30


shape = 327
alpha1 = 20
alpha2 = 10
beta1 = 15
beta2 = 20
delta = 0.001
theta = 20

# 0.导入excel2的数据
excel1 = pd.read_excel("appendix2.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点
coordinates_2 = excel1.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel1.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
# 构建误差向量，0表示该点校正点正常,1表示该校正点可能出现问题
error_vector = list(excel1.values[1:, 5])
# 生成全边集距离矩阵A
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                    coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)

# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量
# 第二次剪枝：减掉方向与AB方向相反的边
# 定义新变量w为字典，键值为(k,i,j)，表示由k点至i点再到j点(i,j)之间的圆弧+切线段的长度
# 首先生成列表triple_edge,元素为元组（k,i,j），表示(k,i)与(i,j)都在可行边集edge里
A_B = [coordinates_2[shape - 1, 0] - coordinates_2[0, 0], coordinates_2[shape - 1, 1] - coordinates_2[0, 1],
       coordinates_2[shape - 1, 2] - coordinates_2[0, 2]]
# edge = copy.copy(edge)
edge2 = copy.copy(edge)
for (i, j) in edge2:
    if A_B[0] * (coordinates_2[j, 0] - coordinates_2[i, 0]) + A_B[1] * (coordinates_2[j, 1] - coordinates_2[i, 1]) + \
            A_B[2] * (coordinates_2[j, 2] - coordinates_2[i, 2]) <= 0:
        edge.remove((i, j))
# 第3次剪枝，以R为参数，AB为轴线排除圆柱外的点,并记录在列表discard中
discard = []
R = 10000
for i in range(shape):
    if A[i, shape - 1] * np.sqrt(1 - (((coordinates_2[shape - 1, 0] - coordinates_2[i, 0]) * (
            coordinates_2[shape - 1, 0] - coordinates_2[0, 0]) + (coordinates_2[shape - 1, 1] - coordinates_2[i, 1]) * (
                                               coordinates_2[shape - 1, 1] - coordinates_2[0, 1]) + (
                                               coordinates_2[shape - 1, 2] - coordinates_2[i, 2]) * (
                                               coordinates_2[shape - 1, 2] - coordinates_2[0, 2])) / (
                                              A[0, shape - 1] * A[i, shape - 1])) ** 2) >= R:
        discard.append(i)

edge1 = copy.copy(edge)
for (i, j) in edge1:
    if i in discard3 or j in discard3:
        edge.remove((i, j))

edge1 = copy.copy(edge)
for (i, j) in edge1:
    if i in discard or j in discard:
        edge.remove((i, j))

triple_edge = set({})
s = 0
for (k, i) in edge:
    for (l, j) in edge:
        if i == l and k != j:
            triple_edge.add((k, i, j))

# 以3-元组triple_edge中的元素为键值，存储夹角信息
angle = {}
for (k, i, j) in triple_edge:
    angle[(k, i, j)] = np.arccos(
        ((coordinates_2[i, 0] - coordinates_2[k, 0]) * (coordinates_2[j, 0] - coordinates_2[i, 0])
         + (coordinates_2[i, 1] - coordinates_2[k, 1]) * (coordinates_2[j, 1] - coordinates_2[i, 1])
         + (coordinates_2[i, 2] - coordinates_2[k, 2]) * (coordinates_2[j, 2] - coordinates_2[i, 2])) / (
                A[k, i] * A[i, j]))
# 以角度angle字典中的值建立新字典distance={}，键值为（k,i,j）
distance = {}
for (k, i, j) in triple_edge:
    distance[(k, i, j)] = 200 * angle[(k, i, j)] + np.sqrt(A[i, j] ** 2 - 400 * A[i, j] * np.sin(angle[(k, i, j)]))

# 2.建立模型
model1 = Model()
# 添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x = model1.addVars(shape, shape, vtype=GRB.BINARY, name='x')
# 添加新变量h[i],v[i],i=0 to shape-1
h = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h")
v = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v")
# 添加啊变量y
y = model1.addVars(triple_edge, vtype=GRB.INTEGER, name="y")
# 添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
# 添加限制条件起始点，中间点，终点的出度入度条件
test1 = [0] * shape  # 出度表达式
test2 = [0] * shape  # 入度表达式
# test1[i]表示i节点的出度
for (i, j) in edge:
    test1[i] = test1[i] + x[i, j]
# test2[i]表示i节点的入度
for (j, i) in edge:
    test2[i] = test2[i] + x[j, i]
for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)
# 添加限制条件对h，v进行约束
for (i, j) in edge:
    model1.addConstr(correction_vector[i] * h[i] + delta * A[i, j] - h[j] <= 10000 - 10000 * x[i, j])
    model1.addConstr((1 - correction_vector[i]) * v[i] + delta * A[i, j] - v[j] <= 10000 - 10000 * x[i, j])
for (k, i, j) in triple_edge:
    model1.addGenConstrAnd(y[(k, i, j)], [x[k, i], x[i, j]], "andconstr")
    model1.addConstr(
        correction_vector[i] * h[i] + delta * distance[(k, i, j)] - h[j] <= 20000 - 10000 * x[i, j] - 10000 * x[k, i])
    model1.addConstr(
        (1 - correction_vector[i]) * v[i] + delta * distance[(k, i, j)] - v[j] <= 20000 - 10000 * x[i, j] - 10000 * x[
            k, i])
for i in range(shape):
    if i == 0:
        model1.addConstr(h[i] == 0)
        model1.addConstr(v[i] == 0)
    elif 0 < i < shape - 1:
        if correction_vector[i] == 1:
            model1.addConstr(h[i] <= alpha2)
            model1.addConstr(v[i] <= alpha1)
        else:
            model1.addConstr(h[i] <= beta2)
            model1.addConstr(v[i] <= beta1)
    else:
        model1.addConstr(h[i] <= theta)
        model1.addConstr(v[i] <= theta)
# 添加目标函数
# 添加罚函数系数
dict_modify = {}
for i in range(shape):
    for j in range(shape):
        dict_modify[i, j] = 10000
dict_modify = tupledict(dict_modify)
distance = tupledict(distance)
test0 = 0
for i in range(shape):
    test0 = test0 + A[0, i] * x[0, i]
model1.setObjective(distance.prod(y) + test0 + 1 * dict_modify.prod(x), GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if np.abs(s.x - 1) < 0.001:
        print('%s''%g' % (s.varName, s.x))
print('Obj:%g' % model1.objVal)
~~~
## 问题二：Excel1绘制最优航迹路线
~~~python
# 导入数据excel1
# 第一题最优路径
node = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]
shape = 613
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx")
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
# 起点为红色，终点为绿色，中间点1为黄色+，0为黄色^
for i in range(613):
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
#使用差分方法模拟圆弧前进，构造圆弧+切线段
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
~~~
## 问题二：Excel2数据集的求解
~~~python
import numpy as np
import pandas as pd
from gurobipy import *

# 导入基本常量
shape = 327
alpha1 = 20
alpha2 = 10
beta1 = 15
beta2 = 20
delta = 0.001
theta = 20
# 0.导入excel2的数据
excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件2：数据集2-终稿.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
# 构建误差向量，0表示该点校正点正常,1表示该校正点可能出现问题
error_vector = list(excel2.values[1:, 5])
# 生成全边集距离矩阵A
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)

# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量
# 第二次剪枝：减掉方向与AB方向相反的边
# 定义新变量w为字典，键值为(k,i,j)，表示由k点至i点再到j点(i,j)之间的圆弧+切线段的长度
A_B = [coordinates_2[shape - 1, 0] - coordinates_2[0, 0], coordinates_2[shape - 1, 1] - coordinates_2[0, 1],
       coordinates_2[shape - 1, 2] - coordinates_2[0, 2]]
for (i, j) in edge:
    if A_B[0] * (coordinates_2[j, 0] - coordinates_2[i, 0]) + A_B[1] * (coordinates_2[j, 1] - coordinates_2[i, 1]) + \
            A_B[2] * (coordinates_2[j, 2] - coordinates_2[i, 2]) <= 0:
        edge.remove((i, j))
# 首先生成列表triple_edge,元素为元组（k,i,j），表示(k,i)与(i,j)都在可行边集edge里
triple_edge = set({})
s = 0
for (k, i) in edge:
    for (l, j) in edge:
        if i == l and k != j:
            triple_edge.add((k, i, j))
# 以3-元组triple_edge中的元素为键值，存储夹角信息
angle = {}
for (k, i, j) in triple_edge:
    angle[(k, i, j)] = np.arccos(
        ((coordinates_2[i, 0] - coordinates_2[k, 0]) * (coordinates_2[j, 0] - coordinates_2[i, 0])
         + (coordinates_2[i, 1] - coordinates_2[k, 1]) * (coordinates_2[j, 1] - coordinates_2[i, 1])
         + (coordinates_2[i, 2] - coordinates_2[k, 2]) * (coordinates_2[j, 2] - coordinates_2[i, 2])) / (
                    A[k, i] * A[i, j]))
# 以角度angle字典中的值建立新字典distance={}，键值为（k,i,j）
distance = {}
for (k, i, j) in triple_edge:
    distance[(k, i, j)] = 200 * angle[(k, i, j)] + np.sqrt(A[i, j] ** 2 - 400 * A[i, j] * np.sin(angle[(k, i, j)]))

#2.建立模型
model1=Model()
#添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x=model1.addVars(shape,shape,vtype=GRB.BINARY,name='x')
#添加新变量h[i],v[i],i=0 to shape-1
h=model1.addVars(shape,vtype=GRB.CONTINUOUS,name="h")
v=model1.addVars(shape,vtype=GRB.CONTINUOUS,name="v")
#添加啊变量y
y=model1.addVars(triple_edge,vtype=GRB.INTEGER,name="y")
#添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i,j]==0:
            model1.addConstr(x[i,j]==0)
#添加限制条件起始点，中间点，终点的出度入度条件
test1=[0]*shape#出度表达式
test2=[0]*shape#入度表达式
#test1[i]表示i节点的出度
for (i,j) in edge:
    test1[i]=test1[i]+x[i,j]
#test2[i]表示i节点的入度
for (j,i) in edge:
    test2[i]=test2[i]+x[j,i]
for i in range(shape):
    if i==0:
        model1.addConstr(test1[i]==1)
        model1.addConstr(test2[i]==0)
    elif 0<i<shape-1:
        model1.addConstr(test1[i]==test2[i])
    else:
        model1.addConstr(test1[i]==0)
        model1.addConstr(test2[i]==1)
#添加限制条件对h，v进行约束
for (i,j) in edge:
    model1.addConstr(correction_vector[i]*h[i]+delta*A[i,j]-h[j]<=10000-10000*x[i,j])
    model1.addConstr((1-correction_vector[i])*v[i]+delta*A[i,j]-v[j]<=10000-10000*x[i,j]) 
for (k,i,j) in triple_edge:
    model1.addGenConstrAnd(y[(k,i,j)],[x[k,i],x[i,j]],"andconstr")
    model1.addConstr(correction_vector[i]*h[i]+delta*distance[(k,i,j)]-h[j]<=20000-10000*x[i,j]-10000*x[k,i])
    model1.addConstr((1-correction_vector[i])*v[i]+delta*distance[(k,i,j)]-v[j]<=20000-10000*x[i,j]-10000*x[k,i])
for i in range(shape):
    if i==0:
        model1.addConstr(h[i]==0)
        model1.addConstr(v[i]==0)
    elif 0<i<shape-1:
        if correction_vector[i]==1:
            model1.addConstr(h[i]<=alpha2)
            model1.addConstr(v[i]<=alpha1)
        else:
            model1.addConstr(h[i]<=beta2)
            model1.addConstr(v[i]<=beta1)
    else:
        model1.addConstr(h[i]<=theta)
        model1.addConstr(v[i]<=theta)      
#添加目标函数
dict_modify={}
for i in range(shape):
    for j in range(shape):
        dict_modify[i,j]=10000
dict_modify=tupledict(dict_modify)
distance=tupledict(distance)
test0=0
for i in range(shape):
    test0=test0+A[0,i]*x[0,i]
model1.setObjective(distance.prod(y)+test0+0*dict_modify.prod(x),GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if s.x==1:
        print('%s''%g'%(s.varName,s.x))
print('Obj:%g'%model1.objVal)
~~~
## 问题二：Excel2绘制最优航迹路线
~~~python
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
~~~
## 问题三：Excel1数据集的求解
~~~python

import numpy as np
import pandas as pd
from gurobipy import *

# 导入第一题的基本常量
shape = 613
alpha1 = 25
alpha2 = 15
beta1 = 20
beta2 = 25
delta = 0.001
theta = 30
# 0.读取excel表1的数据
excel1 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点
coordinates_2 = excel1.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel1.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
# 构建误差向量，1表示该点校正点正常,0表示该校正点可能出现问题
error_vector = list(excel1.values[1:, 5])
# 计算欧式距离，用矩阵A标记
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)
# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量

# 2.建立模型
model1 = Model()
# 添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x = model1.addVars(shape, shape, vtype=GRB.BINARY, name='x')
# 添加新变量h[i],v[i],i=0 to shape-1
h = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h")
h1 = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h1")
v = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v")
v1 = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v1")
# 添加新变量t[i],为0-1变量
t = model1.addVars(shape, vtype=GRB.BINARY, name="k")
# 添加限制条件，当error_vector==0时，t[i]==0
for i in range(shape):
    if error_vector[i] == 0:
        model1.addConstr(t[i] == 0)
# 添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
# 添加限制条件起始点，中间点，终点的出度入度条件
test1 = [0] * shape  # 出度表达式
test2 = [0] * shape  # 入度表达式
# test1[i]表示i节点的出度
for (i, j) in edge:
    test1[i] = test1[i] + x[i, j]
# test2[i]表示i节点的入度
for (j, i) in edge:
    test2[i] = test2[i] + x[j, i]
for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)
for i in range(shape):
    model1.addGenConstrMin(h1[i], [h[i]], 5)
    model1.addGenConstrMin(v1[i], [v[i]], 5)
# 添加限制条件对h，v进行约束
for (i, j) in edge:
    model1.addConstr(
        error_vector[i] * h1[i] * (1 - correction_vector[i]) * (1 - t[i]) + correction_vector[i] * h[i] + delta * A[
            i, j] - h[j] <= 10000 - 10000 * x[i, j])
    model1.addConstr(
        error_vector[i] * v1[i] * correction_vector[i] * (1 - t[i]) + (1 - correction_vector[i]) * v[i] + delta * A[
            i, j] - v[j] <= 10000 - 10000 * x[i, j])
for i in range(shape):
    if i == 0:
        model1.addConstr(h[i] == 0)
        model1.addConstr(v[i] == 0)
    elif 0 < i < shape - 1:
        if correction_vector[i] == 1:
            model1.addConstr(h[i] <= alpha2)
            model1.addConstr(v[i] <= alpha1)
        else:
            model1.addConstr(h[i] <= beta2)
            model1.addConstr(v[i] <= beta1)
    else:
        model1.addConstr(h[i] <= theta)
        model1.addConstr(v[i] <= theta)
    # 添加目标函数
    # 添加罚函数系数
dict_modify = {}
for i in range(shape):
    for j in range(shape):
        dict_modify[i, j] = 10000
dict_modify = tupledict(dict_modify)
test0 = 0
for i in range(shape):
    test0 = test0 + t[i]
model1.setObjective(dict_A.prod(x) + 100000 * test0 + 1 * dict_modify.prod(x), GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if s.x == 1:
        print('%s''%g' % (s.varName, s.x))
print('Obj:%g' % model1.objVal)
~~~
## 问题三：Excel1绘制最优航迹路线
~~~python

node = [0, 503, 69, 506, 371, 183, 194, 450, 286, 485, 612]
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx")
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[612] = 3
fig = plt.figure()
ax = Axes3D(fig)
# 起点，终点为黄色，中间点1为红色+，0为蓝色^
for i in range(613):
    if i == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    elif 0 < i < 612 and correction_vector[i] == 1:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='r', marker='+')
    elif 0 < i < 612 and correction_vector[i] == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='b', marker='^')
    else:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
for i in range(len(node) - 1):
    ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
            [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
            [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]], c='k', linewidth=2)
~~~
## 问题三：Excel2数据集的求解
~~~python

import numpy as np
import pandas as pd
from gurobipy import *

# 导入第一题的基本常量
shape = 327
alpha1 = 20
alpha2 = 10
beta1 = 15
beta2 = 20
delta = 0.001
theta = 20
# 0.读取excel表1的数据
excel2 = pd.read_excel("F:/第十六届华为杯数模比赛/试题/F题/2019年中国研究生数学建模竞赛F题/附件2：数据集2-终稿.xlsx")
# 构建所有点坐标，0为起点，shape-1为终点
coordinates_2 = excel2.values[1:, 1:4]  # 第二列为x坐标，第三列为y坐标，第四列为z坐标
# 构建校正向量，1表示垂直校正点，0表示水平校正点，2表示起点，3表示终点
correction_vector = list(excel2.values[1:, 4])
correction_vector[0] = 2
correction_vector[shape - 1] = 3
# 构建误差向量，1表示该点校正点正常,0表示该校正点可能出现问题
error_vector = list(excel2.values[1:, 5])
# 计算欧式距离，用矩阵A标记
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt(
            (coordinates_2[i, 0] - coordinates_2[j, 0]) ** 2 + (coordinates_2[i, 1] - coordinates_2[j, 1]) ** 2 + (
                        coordinates_2[i, 2] - coordinates_2[j, 2]) ** 2)
# 将A转化成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 1.剪枝过程
# V为垂直校正点集合，H为水平校正点集合.共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass
C = np.ones((shape, shape))
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] > min(beta1, beta2) / delta:
            dict_A[i, j] = 0
            C[i, j] = 0
for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta / delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0
for i in range(shape):
    C[i, i] = 0
edge = []  # 边集
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass
# NOTICE：以上剪枝没有减掉所有以0为终点或者以shape-1为起点的变量

# 2.建立模型
model1 = Model()
# 添加新变量x[i,j],i=0 to shape-1,j=0 to shape-1
x = model1.addVars(shape, shape, vtype=GRB.BINARY, name='x')
# 添加新变量h[i],v[i],i=0 to shape-1
h = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h")
h1 = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="h1")
v = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v")
v1 = model1.addVars(shape, vtype=GRB.CONTINUOUS, name="v1")
# 添加新变量t[i],为0-1变量
t = model1.addVars(shape, vtype=GRB.BINARY, name="k")
# 添加限制条件，当error_vector==0时，t[i]==0
for i in range(shape):
    if error_vector[i] == 0:
        model1.addConstr(t[i] == 0)
# 添加限制条件 x[i,j]==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
# 添加限制条件起始点，中间点，终点的出度入度条件
test1 = [0] * shape  # 出度表达式
test2 = [0] * shape  # 入度表达式
# test1[i]表示i节点的出度
for (i, j) in edge:
    test1[i] = test1[i] + x[i, j]
# test2[i]表示i节点的入度
for (j, i) in edge:
    test2[i] = test2[i] + x[j, i]
for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)
for i in range(shape):
    model1.addGenConstrMin(h1[i], [h[i], 5])
    model1.addGenConstrMin(v1[i], [v[i], 5])
# 添加限制条件对h，v进行约束
for (i, j) in edge:
    model1.addConstr(
        error_vector[i] * h1[i] * (1 - correction_vector[i]) * (1 - t[i]) + correction_vector[i] * h[i] + delta * A[
            i, j] - h[j] <= 10000 - 10000 * x[i, j])
    model1.addConstr(
        error_vector[i] * v1[i] * correction_vector[i] * (1 - t[i]) + (1 - correction_vector[i]) * v[i] + delta * A[
            i, j] - v[j] <= 10000 - 10000 * x[i, j])
    # model1.addConstr(error_vector[i]*h1[i]*(1-correction_vector[i])*(1-t[i])+correction_vector[i]*h[i]+delta*A[i,j]-h[j]<=10000-10000*x[i,j])
    # model1.addConstr(error_vector[i]*v1[i]*correction_vector[i]*(1-t[i])+(1-correction_vector[i])*v[i]+delta*A[i,j]-v[j]<=10000-10000*x[i,j])
for i in range(shape):
    if i == 0:
        model1.addConstr(h[i] == 0)
        model1.addConstr(v[i] == 0)
    elif 0 < i < shape - 1:
        if correction_vector[i] == 1:
            model1.addConstr(h[i] <= alpha2)
            model1.addConstr(v[i] <= alpha1)
        else:
            model1.addConstr(h[i] <= beta2)
            model1.addConstr(v[i] <= beta1)
    else:
        model1.addConstr(h[i] <= theta)
        model1.addConstr(v[i] <= theta)
    # 添加目标函数
    #添加罚函数
dict_modify = {}
for i in range(shape):
    for j in range(shape):
        dict_modify[i, j] = 10000
dict_modify = tupledict(dict_modify)
test0 = 0
for i in range(shape):
    test0 = test0 + t[i]
model1.setObjective(dict_A.prod(x) + 100000 * test0 + 1 * dict_modify.prod(x), GRB.MINIMIZE)
model1.update()
model1.optimize()
model1.write("model1_without_turning.lp")
for s in model1.getVars():
    if s.x == 1:
        print('%s''%g' % (s.varName, s.x))
print('Obj:%g' % model1.objVal)
~~~
## 问题三：Excel2绘制最优航迹路线
~~~python

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
~~~
