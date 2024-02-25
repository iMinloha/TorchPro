'''
@auther Minloha
@date 2023-02-24
@:exception 卷积神经网络
'''
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms

# 训练集
train_loader = torch.utils.data.DataLoader(
    # 第一个参数是数据集的路径, 可以修改
    # download=True: 如果数据集不存在则下载
    datasets.MNIST('data', train=True, download=True,
                   # transform: 对数据集进行预处理，这里是将数据集转换为tensor并归一化
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    # batch_size: 每次读取的数据个数(512大概占显存1G)
    # shuffle: 是否打乱数据集
    batch_size=512, shuffle=True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    # 降低测试时间
    batch_size=256, shuffle=True)


# 评估模型，计算正确率
# 正确率 = 正确预测的个数 / 总预测的个数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    # 正确预测的个数, 总预测的个数
    acc_sum, n = 0.0, 0
    # 不记录梯度
    with torch.no_grad():
        # 遍历数据集
        for X, y in data_iter:
            # 判断数据类型正确
            if isinstance(net, nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                # 计算正确率(使用CPU计算,降低GPU负担)
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型,不考虑GPU
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    # 计算所有预测正确的个数
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            # 计算总预测的个数
            n += y.shape[0]
    return acc_sum / n


# 训练模型
'''
:param net: 神经网络
:param train_iter: 训练数据集
:param num_epochs: 训练次数
:param lr: 学习率
:param device: 设备
'''


def train_ch6(net, train_iter, num_epochs, lr, device):
    net.to(device)
    print("training on", device)
    # Adam，当然还有其他的比如SGD等等
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 交叉熵损失函数(MSE也可以)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start))


# 定义一个简单的卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # mnist是一张灰度图，所以输入通道数为1，长宽为28
            # 一次一张，一张一个通道，长宽为28[1, 1, 28, 28]
            # Conv2d四个参数，第一个参数是输入通道数，第二个参数是输出通道数，第三个参数是卷积核大小，第四个参数是步长
            nn.Conv2d(1, 32, 3, 1),
            # 输出通道数为32，长宽为26，计算方式为(28-3+1)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 池化层，长宽除以2并向上取证，输出通道数不变
            # output shape: (32, 13, 13)
            nn.Conv2d(32, 64, 3, 1),
            # output shape: (64, 11, 11)
            # 输出通道数为64，长宽为11，计算方式为(13-3+1)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output shape: (64, 5, 5)
            # 一样的计算方法
            nn.Flatten(),
            # 取出所有参数，64个输出，每个输出都是长宽为5的矩阵
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# 测试模型
def test(model, device, test_loader):
    # 切换模型为评估模式（不启用Dropout）
    model.eval()
    test_loss = 0
    correct = 0
    # 关闭梯度记录
    with torch.no_grad():
        # 遍历测试集
        for data, target in test_loader:
            # 全放在GPU
            data, target = data.to(device), target.to(device)
            # 计算输出
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()  # 对预测正确的数据个数进行累加

    # 计算损失平均值
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    model = CNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ch6(model, train_loader, 5, 0.001, device)
    test(model, device, test_loader)
