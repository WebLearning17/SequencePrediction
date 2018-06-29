import torch
from torch import nn,optim
from torch.autograd import  Variable

#查阅资料了解init 的作用
from torch.nn import init

#已知1949年-1960年的飞机客流量

#获取客流量
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv',usecols=[1])

print(data)


#定义numpy 转换成Tensor的函数
toTs=lambda x:torch.from_numpy(x)

#数据预处理
data=data.dropna()
dataSet=data.values
dataSet=dataSet.astype('float32')
print("data Shape:",dataSet.shape)

#数据归一化
def MinMaxScaler(X):
    mx,mi=np.max(X),np.min(X)
    X_std=(X-mx)/(mx-mi)
    return X_std

dataSet=MinMaxScaler(dataSet)

#将数据分为训练集合测试集
train=dataSet[:12*10]
real=dataSet

input_size= 3
