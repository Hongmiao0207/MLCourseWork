import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# =============1.数据处理部分 ===========
dataset = pd.read_csv('../../Resource/SeoulBikeData.csv', encoding="ISO-8859-1")
print(dataset.shape)

# 处理字符
season_dict = dataset['Seasons'].unique().tolist()
date_dict = dataset['Date'].unique().tolist()
holiday_dict = dataset['Holiday'].unique().tolist()
functioningDay_dict = dataset['Functioning Day'].unique().tolist()

dataset['Seasons'] = dataset['Seasons'].apply(lambda x: season_dict.index(x))
dataset['Date'] = dataset['Date'].apply(lambda x: date_dict.index(x))
dataset['Holiday'] = dataset['Holiday'].apply(lambda x: holiday_dict.index(x))
dataset['Functioning Day'] = dataset['Functioning Day'].apply(lambda x: functioningDay_dict.index(x))

# print(dataset)
X = dataset.drop(['Rented Bike Count'], axis=1).values
y = dataset.iloc[:, 1].values
# print(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)


# =============2.回归部分 =============
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
# =============3.1 决策树回归 =============
from sklearn import tree

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

# =============3.2 线性回归 =============
from sklearn import linear_model

model_LinearRegression = linear_model.LinearRegression()

# =============3.3 SVM回归 =============
from sklearn import svm

model_SVR = svm.SVR(C=1e3, gamma=0.01)

# =============3.4 KNN回归 =============
from sklearn import neighbors

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()

# =============3.5 随机森林回归 =============
from sklearn import ensemble

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树

# # =============3.6 Adaboost回归 =============
# from sklearn import ensemble
#
# model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
#
# # =============3.7 GBRT回归 =============
# from sklearn import ensemble
#
# model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
#
# # =============3.8 Bagging回归 =============
# from sklearn.ensemble import BaggingRegressor
#
# model_BaggingRegressor = BaggingRegressor()
#
# # =============3.9 ExtraTree极端随机树回归 =============
# from sklearn.tree import ExtraTreeRegressor
#
# model_ExtraTreeRegressor = ExtraTreeRegressor()

# =============3.10 神经网络 =============
from sklearn.neural_network import MLPRegressor

model_MLPR = MLPRegressor(alpha=1e-5, max_iter=2000)

# =============4.具体方法调用部分 =============
try_different_method(model_DecisionTreeRegressor)
try_different_method(model_LinearRegression)
try_different_method(model_SVR)
try_different_method(model_KNeighborsRegressor)
try_different_method(model_RandomForestRegressor)
# try_different_method(model_AdaBoostRegressor)
# try_different_method(model_GradientBoostingRegressor)
# try_different_method(model_BaggingRegressor)
# try_different_method(model_ExtraTreeRegressor)
try_different_method(model_MLPR)

