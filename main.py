import pandas
import numpy
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, SGDRegressor
from matplotlib import pyplot


def load_data():
    data = numpy.loadtxt('houses.txt', delimiter=',')
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1]
    return X, y

# 从txt中加载数据
X_train, y_train = load_data()
X_features = ['sqft', 'bedrooms', 'floors', 'age']
# Z-score标准化
scaler = preprocessing.StandardScaler()
X_zsnorm = scaler.fit_transform(X_train)
# 实例化一个随机梯度下降对象
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_zsnorm, y_train)
# 梯度下降计算出的系数和截距，只做展示，预测时直接使用sklearn提供的方法
w_norm = sgdr.coef_
b_norm = sgdr.intercept_
# 使用sklearn的方法计算出预测值
y_pred = sgdr.predict(X_zsnorm)    #也可以使用numpy.dot(X_norm, w_norm) + b_norm来计算预测值
# 绘图
fig, ax = pyplot.subplots(1, 4, sharey=True, figsize=(13, 4))
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, color='blue', label='target')
    ax[i].scatter(X_train[:, i], y_pred, color='orange', label='predict')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('Price')
ax[0].legend()
fig.suptitle('target versus prediction using z-score normalized model')    # pyplot.suptitle()也行
pyplot.show()