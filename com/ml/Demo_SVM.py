from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../../Resource/DATA.csv', sep=';')

X = dataset.iloc[1:, 3: -2].values
y = dataset.iloc[1:, -1].values
# print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

svm_c = svm.SVC()
svm_c.fit(X_train, y_train)
print(svm_c.score(X_test, y_test))
