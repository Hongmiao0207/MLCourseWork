import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as exp_gra
from sklearn.model_selection import train_test_split
import graphviz

dataset = pd.read_csv('../../Resource/DATA.csv', sep=';')

X = dataset.iloc[1:, 3:-2].values
y = dataset.iloc[1:, -1].values
# print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

clf = DecisionTreeClassifier(criterion="entropy"
                             , random_state=30
                             , splitter="random"
                             , max_depth=9
                             , min_samples_leaf=2
                             )
clf.fit(X_train, y_train)

# Predict
print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))

# dot_data = exp_gra(clf,
#                    feature_names=["Age", "Sex", "school type", "Scholarship type", "Additional work", "activity",
#                                   "partner", "salary", "Transportation", "Accommodation", "Mother Edu", "Father Edu",
#                                   "sisters/brothers", "Parental status", "M occupation", "F occupation", "study hours", "Reading frequency1",
#                                   "Reading frequency2", "seminars", "projects", "Attendance", "Preparation to exam1", "Preparation to exam2",
#                                   "notes", "Listening", "Discussion", "Flip-classroom", "last grade point", "grade point"],
#                    filled=True,
#                    rounded=True)
#
# graph = graphviz.Source(dot_data)
# graph.view()
