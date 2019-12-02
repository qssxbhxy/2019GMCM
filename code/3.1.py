
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
