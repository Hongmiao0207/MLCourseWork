from sklearn import neural_network
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('../../../../Resource/tae.data', header=None)

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values


# 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
#
#
def test_validate(x_test, y_test, y_predict, classifier):
    x = range(len(y_test))
    plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"true_v")
    plt.plot(x, y_predict, "go", markersize=8, zorder=2, label=u"predict_v,$R$=%.3f" % classifier.score(x_test, y_test))
    plt.legend(loc="upper left")
    plt.xlabel("number")
    plt.ylabel("true?")
    plt.show()


# 神经网络数字分类
def multi_class_nn():
    # 对数据的训练集进行标准化
    ss = StandardScaler()
    x_regular = ss.fit_transform(X)
    # 划分训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x_regular, y, test_size=0.3, random_state=36)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1, max_iter=2000)
    clf.fit(x_train, y_train)
    # 模型效果获取
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    print(accuracy_score(y_predict, y_test))
    # 绘制测试集结果验证
    test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=clf)


multi_class_nn()

# nn_c = neural_network.MLPClassifier()
# nn_c.fit(X_train, y_train)
# print(nn_c.score(X_test, y_test))
