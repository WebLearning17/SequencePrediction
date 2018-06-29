import torch
from torch import nn,optim
from torch.autograd import Variable
from  torch.nn import init
import numpy as np
from seqInit import toTs
from seqInit import input_size,train,real

#定义LSTM 模型
class lstmModel(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,layer_num):
        super().__init__()
        self.lstmLayer=nn.LSTM(in_dim,hidden_dim,layer_num)
        self.relu=nn.ReLU()
        self.fcLayer=nn.Linear(hidden_dim,out_dim)

        self.weightInit=(np.sqrt(1.0/hidden_dim))

    def forward(self, x):
        out,_=self.lstmLayer(x)
        s,b,h=out.size() #seq,batch,hidden
        #out=out.view(s*b,h)
        out=self.relu(out)
        out=out[12:]
        out=self.fcLayer(out)
        #out=out.view(s,b,-1)

        return out

    #初始化权重
    def weightInit(self, gain=1):
            # 使用初始化模型参数
        for name, param in self.named_parameters():
            if 'lstmLayer.weight' in name:
                init.orthogonal(param, gain)

# 输入维度为1，输出维度为1，隐藏层维数为5, 定义LSTM层数为2
lstm = lstmModel(1, 5, 1, 2)

# 定义损失函数和优化函数
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr = 1e-2)

# 处理输入

train = train.reshape(-1, 1, 1)
x = torch.from_numpy(train[:-1])
y = torch.from_numpy(train[1:])[12:]
print(x.shape, y.shape)

# 训练模型

frq, sec = 3500, 350
loss_set = []

for e in range(1, frq + 1) :
    inputs = Variable(x)
    target = Variable(y)
    #forward
    output = lstm(inputs)
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


lstm = lstm.eval()
# 预测结果并比较

px = real[:-1].reshape(-1, 1, 1)
px = torch.from_numpy(px)
ry = real[1:].reshape(-1)
varX = Variable(px)
py = lstm(varX).data
py = np.array(py).reshape(-1)
print(px.shape, py.shape, ry.shape)
np.save('prediction',py[-36:])
np.save('real',ry[-36:])


# # 画出实际结果和预测的结果
# plt.plot(py[-24:], 'r', label='prediction')
# plt.plot(ry[-24:], 'b', label='real')
# plt.legend(loc='best')