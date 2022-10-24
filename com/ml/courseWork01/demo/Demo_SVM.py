import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import classification_report

# datasetColumn = pd.read_csv('../../Resource/tae.names')
dataset = pd.read_csv('../../../../Resource/tae.data', header=None)

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

# 数据标准化
stdScaler = StandardScaler().fit(X_train)
X_trainStd = stdScaler.transform(X_train)
X_testStd = stdScaler.transform(X_test)

# 建立模型
model = SVC(kernel='rbf', probability=True)
model.fit(X_trainStd, y_train)
pre = model.predict(X_testStd)
pre1 = model.predict(X_trainStd)
# print(model.score(X_testStd, y_test))

# print(classification_report(y_test, pre))

print(y.shape, y_test.shape, y_train.shape, pre.shape)
print('使用SVM预测数据的准确率为：', accuracy_score(y_train, pre1))
print('使用SVM预测数据的准确率为：', accuracy_score(y_test, pre))
