import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as exp_gra
from sklearn.model_selection import train_test_split
import graphviz

# 读取数据
dataset = pd.read_csv('../../Resource/tae.data', header=None)

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

# clf = DecisionTreeClassifier(criterion="entropy"
#                              , random_state=30
#                              , splitter="random"
#                              , max_depth=9
#                              , min_samples_leaf=2
#                              )
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict
print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))

dot_data = exp_gra(clf,
                   feature_names=["Native English or not", "Instructor", "Course", "Semester", "Class size"],
                   filled=True,
                   rounded=True)

graph = graphviz.Source(dot_data)
graph.view()
