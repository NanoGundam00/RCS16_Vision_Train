```
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义一个多层感知器（MLP）类，继承自 nn.Module
class MLP(nn.Module):
    #构造函数初始化数据，创建类的实例的时候需要输入这三个参数
    def __init__(self, input_size, hidden_size, output_size):
        #在构造函数中调用父类（nn.Module）的构造函数，确保正确地初始化父类的属性。
        super(MLP, self).__init__()

        # 构建网络结构
        #self.layers是一个nn.Sequential类的对象，
        #layers是自己创造的一个属性，self类似c++中的this指针
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),     #定义了一个线性层，其输入大小为 input_size，输出大小为 hidden_size
            nn.ReLU(),                              #是一个 ReLU 激活函数层，不改变大小
            nn.Linear(hidden_size, output_size)     #定义了另一个线性层，其输入大小为 hidden_size，输出大小为 output_size
        )

    def forward(self, x):
        # 定义前向传播
        #对输入张量 x 进行形状变换，将其转换为二维张量，其中第一维保留样本数量，第二维自动计算得到以保持原有数据个数的维度。
        #这一步操作通常被称为"Flatten"，用于将多维数据展平为一维数据。
        x = x.view(x.size(0), -1) # Flatten the input tensor
        #将flatten后的张量，通过之前在 __init__ 方法中创建的 layers 模型进行前向传播计算
        out = self.layers(x)
        return out

# 超参数设置
#输入层的大小为 784，对应于输入数据的特征数量
input_size = 784
#隐藏层的大小为 128，表示网络中间的隐藏单元数量
hidden_size = 128
#输出层的大小为 10，对应于分类问题中的类别数量
output_size = 10
#学习率为 0.001，控制着在优化过程中权重更新的步幅大小
learning_rate = 0.001
#批处理大小为 64，表示每次迭代中所使用的样本数量
batch_size = 64
#训练进行的总轮数为 10，表示将对整个训练数据集进行 10 次遍历
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),   #将输入数据转换为张量格式。将输入数据的值从范围 [0, 255] 归一化到 [0, 1]，并将数据类型转换为 PyTorch 中的张量类型
    transforms.Normalize((0.1307,), (0.3081,))  #对数据进行归一化处理，对每个通道的数据进行均值和方差归一化
])

# 加载MNIST数据集
#root 参数指定了数据集保存的路径，train=True 表示加载训练集，train=False 表示加载测试集，download=True 表示如果数据集不存在，则自动从官方网站下载数据集。
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#使用 DataLoader 类将数据集包装成可迭代的数据加载器。
# batch_size 参数指定了每个批次的样本数量，shuffle=True 表示在每个轮次中随机打乱样本的顺序，shuffle=False 表示使用原始顺序加载样本。
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型、损失函数和优化器对象
#使用了一个自定义的 MLP 类来创建模型对象
model = MLP(input_size, hidden_size, output_size)
#定义了交叉熵损失函数对象，用于计算模型输出和真实标签之间的损失
criterion = nn.CrossEntropyLoss()
#定义了优化器对象，使用 Adam 优化算法来更新模型参数。model.parameters() 用于获取模型的可学习参数，lr 参数指定了初始学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        images=images
        labels= labels
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        #将优化器中的梯度缓存清零，以避免之前计算得到的梯度对当前步骤的梯度更新产生影响
        optimizer.zero_grad()
        #根据当前的损失值，通过自动求导（Autograd）计算出参数的梯度。损失值通过反向传播的过程，将梯度信息传递到各个参数
        loss.backward()
        #根据计算得到的梯度更新模型的参数。通过优化器调用该方法，可以根据指定的优化算法以及学习率等参数，更新模型中的参数值
        optimizer.step()

        # 输出损失信息
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 测试模型
#将模型切换到评估模式。在评估过程中，以便在计算过程中不进行梯度计算和参数更新。这通常用于禁用一些具有随机性的操作
model.eval()
#用于记录预测正确的样本数和总样本数
correct = 0
total = 0
#创建一个环境，在这个环境中的计算不会被记录梯度。在评估阶段，我们通常不需要计算梯度，而且禁用梯度计算可以提高计算效率
with torch.no_grad():
    for images, labels in test_loader:
        #使用模型对输入 images 进行预测，得到预测的输出
        outputs = model(images)
        #将预测的输出张量中的最大值和最大值所在的索引提取出来。predicted 是预测的类别标签
        _, predicted = torch.max(outputs.data, 1)
        #增加总样本数，即将当前批次中的样本个数
        total += labels.size(0)
        #将预测标签 predicted 和真实标签 labels 逐元素进行比较，相同则为预测正确，然后将预测正确的个数累加到 correct 中
        correct += (predicted == labels).sum().item()

#用pt格式保存到本地
torch.save(model,"MLP_model.pt")
#输出准确率
print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")

```