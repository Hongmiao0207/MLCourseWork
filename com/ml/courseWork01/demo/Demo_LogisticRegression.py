import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import load_breast_cancer


# 读取数据
dataset = pd.read_csv('../../../../Resource/tae.data', header=None)

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)


# --------------------------方法1--------------------------------------------------------------
# lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
# lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)
#
#
# lrl1 = lrl1.fit(X, y)
# # # 逻辑回归的重要属性 coef_，查看每个特征所对应的参数
# # print(lrl1.coef_)
# # # 查看不为0的系数有多少个
# # print(((lrl1.coef_ != 0).sum(axis=1)))
# print(lrl1.score(X_test, y_test))
#
# lrl2 = lrl2.fit(X, y)
# # print(lrl2.coef_)
# # print(lrl2.coef_ != 0)
# print(lrl2.score(X_test, y_test))

# 对比可知， l1正则化会把参数压缩到0，l2正则化会尽量让每一个参数都对模型有贡献


# ------------------------------------方法2------------------------------------------------------
# 不求解参数的模型，就没有损失函数
# 逻辑回归需要求解出参数来建立模型
# 正则化，防止数据过拟合

# l1 = []
# l2 = []
# l1_test = []
# l2_test = []
#
# for i in np.linspace(0.01, 0.8, 19):
#     lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
#     lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
#
#     lrl1 = lrl1.fit(X_train, y_train)
#
#     l1.append(accuracy_score(lrl1.predict(X_train), y_train))
#     l1_test.append(accuracy_score(lrl1.predict(X_test), y_test))
#
#     lrl2 = lrl2.fit(X_train, y_train)
#
#     l2.append(accuracy_score(lrl2.predict(X_train), y_train))
#     l2_test.append(accuracy_score(lrl2.predict(X_test), y_test))
#
# graph = [l1, l2, l1_test, l2_test]
# color = ["green", "black", "lightgreen", "gray"]
# label = ["L1", "L2", "L1_test", "L2_test"]
#
# plt.figure(figsize=(6, 6))
# for i in range(len(graph)):
#     plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
#
# plt.legend()
# plt.show()
#
# print(l1_test)
# print(l2_test)
#
# log_reg = LR()
# log_reg.fit(X_train, y_train)
# print(log_reg.score(X_train, y_train))
# print(log_reg.score(X_test, y_test))


# ---------------------方法3---------------------------------
def PolynomialLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])


poly_log_reg3 = PolynomialLogisticRegression(degree=20, C=0.1)
poly_log_reg3.fit(X_train, y_train)

print(poly_log_reg3.score(X_train, y_train))

print(poly_log_reg3.score(X_test, y_test))
