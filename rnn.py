import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.nn import init

#引入初始化相关的内容
from  seqInit import toTs
from  seqInit import input_size
from seqInit import train,real

import numpy as np

#定义RNN 模型
class rnnModel(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,layer_num):
        super().__init__()
        self.rnnLayer=nn.RNN(in_dim,hidden_dim,layer_num)
        self.fcLayer=nn.Linear(hidden_dim,out_dim)
        optim_range=np.sqrt(1.0/hidden_dim)
        self.weightInit(optim_range)

    def forward(self, x):
        out,_=self.rnnLayer(x)
        out=out[12:]
        out=self.fcLayer(out)
        return  out

    def weightInit(self,gain=1):
        #使用初始化模型参数
        for name ,param in self.named_parameters():
            if 'rnnLayer.weight' in name:
                init.orthogonal(param,gain)


#输入维度1，输出维度1 ，隐藏层1，定义rnn层数2

rnn=rnnModel(1,10,1,2)

#确定损失函数和优化函数
criterion=nn.MSELoss()
optimizer=optim.Adam(rnn.parameters(),lr=1e-2)

# 处理输入

def create_dataset(dataset) :
    data = dataset.reshape(-1, 1, 1)
    return torch.from_numpy(data)

trainX = create_dataset(train[:-1])
trainY = create_dataset(train[1:])[12:]
print(trainX.shape, trainY.shape)

# 训练RNN模型
frq, sec = 2000, 200
loss_set = []
for e in range(1, frq + 1) :
    inputs = Variable(trainX)
    target = Variable(trainY)
    # forward
    output = rnn(inputs)
    loss = criterion(output, target)
    # update gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print training information
    print_loss = loss.data[0]
    loss_set.append((e, print_loss))
    if e % sec == 0 :
        print('Epoch[{}/{}], loss = {:.5f}'.format(e, frq, print_loss))

# 测试

rnn = rnn.eval()
px = real[:-1].reshape(-1, 1, 1)
px = torch.from_numpy(px)
ry = real[1:].reshape(-1)
varX = Variable(px, volatile=True)
py = rnn(varX).data
py = np.array(py).reshape(-1)
print(px.shape, py.shape, ry.shape)
np.save('prediction',py[-24:])
np.save('real',ry[-24:])

# # 画出实际结果和预测的结果
# plt.plot(py[-24:], 'r', label='prediction')
# plt.plot(ry[-24:], 'b', label='real')
# plt.legend(loc='best')
