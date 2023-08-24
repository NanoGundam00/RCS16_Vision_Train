```
import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

#数据准备
np.random.seed(24)
x = np.random.rand(100,1)
y = x * 5 + np.random.randn(100,1) * 0.3
x=torch.from_numpy(x).float()
y=torch.from_numpy(y).float()

#设置参数
epochs = 100
learning_rate = 0.01
batch_size = 7
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# #画出图像
# plt.scatter(x,y,marker='+')
# plt.xlabel("Qi", color = 'red')
# plt.show()

#加载数据
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

Loss = []  # 损失值列表

for epoch in range(1, epochs + 1):
    sum_loss = 0
    for batch_id, (bx, by) in enumerate(dataloader):
        # 预测
        pre = bx * w + b
        # 计算均方差
        loss = torch.mean((pre - by) ** 2)
        sum_loss += loss.item()
        # 反向传播
        loss.backward()
        w.data -= learning_rate * w.grad.data
        b.data -= learning_rate * b.grad.data
        #注意梯度清零的顺序，由于一开始是没有梯度的，所以要将梯度清零的函数写在最后
        w.grad.zero_()
        b.grad.zero_()

    Loss.append(sum_loss)

#画图
plt.subplot(121)
Loss_x=[i for i in range(1,epochs+1)]
plt.plot(Loss_x,Loss)
fig1 = plt.xlabel("Zhitong Qi",color ="g")


#获取一个标量张量中的元素值，并将其转换成Python原生的标量类型
plt.subplot(122)
w=w.item()
b=b.item()
xx=np.linspace(0,1,100)
h=w * xx+ b
plt.plot(xx,h)
fig2 = plt.scatter(x,y,marker='+',color='red')
plt.xlabel("Zhitong Qi",color ="g")

plt.show()

```