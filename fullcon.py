'''
@auther Minloha
@date 2023-02-24
@:exception 全连接神经网络
'''
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 全连接神经网络
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        # 一层全连接层，输入1，输出1
        # 输入层
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        # 前传
        x = self.layer1(x)
        return x


# 生成数据
def generate_data():
    # 生成100个随机数, 1列
    x = torch.rand(100, 1)
    # 生成y = 3.3x + 2.2 + 一些噪声
    y = x * 3.3 + 2.2 + torch.rand(100, 1)
    return x, y


# 生成二次函数数据
def generate_data2():
    # 生成100个随机数, 1列
    x = torch.rand(100, 1)
    # 生成y = 3.3x^2 + 2.2 + 一些噪声
    y = x * x * 3.3 + 2.2 + torch.rand(100, 1)
    return x, y


# 训练
def train(net, x, y, optimizer, loss):
    # 训练1000次
    for i in range(1000):
        y_hat = net(x)
        # 计算损失
        l = loss(y_hat, y)
        optimizer.zero_grad()
        # 反传
        l.backward()
        optimizer.step()
        # 每100次输出一次损失
        if i % 100 == 0:
            print('epoch %d, loss %.4f' % (i, l.item()))
    return net


if __name__ == "__main__":
    # 生成数据
    x, y = generate_data()
    net = FNN()
    # 损失为均方误差
    loss = nn.MSELoss()
    # 优化器为随机梯度下降，lr为学习率
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    net = train(net, x, y, optimizer, loss)
    # 顺便画个图
    plt.scatter(x, y)
    # 画出拟合的直线
    plt.plot(x, net(x).detach().numpy(), 'r-')
    # 展示！
    plt.show()
