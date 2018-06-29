# 引入torch相关模块
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init

# 引入初始化文件中的相关内容
from seqInit import toTs
from seqInit import input_size
from seqInit import train, real

# 引入画图工具
import numpy as np


# 定义GRU模型

class gruModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layer):
        super().__init__()
        self.gruLayer = nn.GRU(in_dim, hidden_dim, hidden_layer)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.gruLayer(x)
        out = out[12:]
        out = self.fcLayer(out)
        return out


# 输入维度为1，输出维度为1，隐藏层维数为5, 定义LSTM层数为2
gru = gruModel(1, 5, 1, 2)

# 定义损失函数和优化函数

criterion = nn.MSELoss()
optimizer = optim.Adam(gru.parameters(), lr=1e-2)

# 处理输入

train = train.reshape(-1, 1, 1)
x = torch.from_numpy(train[:-1])
y = torch.from_numpy(train[1:])[12:]
print(x.shape, y.shape)

# 训练模型

frq, sec = 4000, 400
loss_set = []

for e in range(1, frq + 1) :
    inputs = Variable(x)
    target = Variable(y)
    #forward
    output = gru(inputs)
    loss = criterion(output, target)
    # update paramters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print training information
    print_loss = loss.item()
    loss_set.append((e, print_loss))
    if e % sec == 0 :
        print('Epoch[{}/{}], Loss: {:.5f}'.format(e, frq, print_loss))

gru = gru.eval()

# 预测结果并比较

px = real[:-1].reshape(-1, 1, 1)
px = torch.from_numpy(px)
ry = real[1:].reshape(-1)
varX = Variable(px)
py = gru(varX).data
py = np.array(py).reshape(-1)
print(px.shape, py.shape, ry.shape)

np.save('prediction',py[-36:])
np.save('real',ry[-36:])
