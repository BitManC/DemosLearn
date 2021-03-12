import numpy as np
import math


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):

    '''
    data: stack data
    t: 第几个交易日
    n: 目前理解为回测天数，或者周期，可以是个超参数

    这里如果可以构造复杂些，可以用 GCN 或者 其他神经网络来构造
    '''
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0, n < t 时，表示才开始 window 大小内的天数
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i])) # 对股票数据差分做 sigmod

    return np.array([res])
