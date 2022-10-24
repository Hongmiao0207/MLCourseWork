import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

lr = LinearRegression()
lr.fit(X_test, y_test)
y_pred = lr.predict(X_test).astype('int')
print('Coefficients: \n', lr.coef_)
print('y_pre', y_pred)
print('y_test', y_test)
print('分数', lr.score(X_test, y_test))

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

lr_r2 = r2_score(y_test, y_pred)
print("R2 is", lr_r2)

# 用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 交叉验证
predicted = cross_val_predict(lr, X, y, cv=50)
# 用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))

