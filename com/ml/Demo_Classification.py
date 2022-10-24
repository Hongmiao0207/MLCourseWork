import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ==================1. 读取数据 =================
dataset = pd.read_csv('../../Resource/tae.data', header=None)

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

# 数据标准化
stdScaler = StandardScaler().fit(x_train)
x_train = stdScaler.transform(x_train)
x_test = stdScaler.transform(x_test)


# =============2. 分类部分 =============

def try_different_method(model):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


# =============3.具体方法选择 =============

# =============3.1 决策树分类 =============
from sklearn.tree import DecisionTreeClassifier

model_DecisionTreeClassifier = DecisionTreeClassifier()

# =============3.2 逻辑回归 =============
from sklearn.linear_model import LogisticRegression

model_LogisticRegression = LogisticRegression(solver='liblinear', max_iter=200)

# =============3.3 线性SVC =============
from sklearn.svm import LinearSVC

model_LinearSVC = LinearSVC(max_iter=2000, C=2)

# =============3.4 SVC =============
from sklearn.svm import SVC

model_SVC = SVC(kernel='rbf', probability=True)

# =============3.5 神经网络 =============
from sklearn.neural_network import MLPClassifier

model_MPLC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2250,), max_iter=2000)

# =============4.具体方法调用部分 =============
try_different_method(model_DecisionTreeClassifier)
try_different_method(model_LogisticRegression)
try_different_method(model_LinearSVC)
try_different_method(model_SVC)
try_different_method(model_MPLC)
