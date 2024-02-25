'''
@auther Minloha
@date 2023-02-25
@:exception 循环神经网络
'''
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "serial.csv"
TRAIN = 0.7  # 训练数据占比


def getData():
    df = pd.read_csv(file, usecols=[1])
    data_csv = df.dropna()
    dataset = data_csv.values
    dataset = dataset.astype('float32')
    scalar = np.max(dataset) - np.min(dataset)
    dataset = list(map(lambda x: x / scalar, dataset))
    return dataset


'''
:param dataset 数据集
:param step 预测步长
'''


def create_dataset(dataset, step=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - step):
        a = dataset[i:(i + step)]
        dataX.append(a)
        dataY.append(dataset[i + step])
    return torch.tensor(np.array(dataX)), torch.tensor(np.array(dataY))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.net = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.net(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.net = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.net(x)
        out = self.fc(out[:, -1, :])
        return out


def train(net, x, y, optimizer, loss):
    for i in range(80):
        y_hat = net(x)
        # 计算损失
        l = loss(y_hat, y)
        optimizer.zero_grad()
        # 反传
        l.backward()
        optimizer.step()
        # 每10次输出一次损失
        if i % 10 == 0:
            print('epoch %d, loss %.4f' % (i, l.item()))


def test(net, x, y, loss):
    y_hat = net(x)
    l = loss(y_hat, y)
    plt.plot(y.numpy(), label='real')
    plt.plot(y_hat.detach().numpy(), label='predict')
    plt.legend()
    plt.show()
    print('test loss %.4f' % (l.item()))


def RNNtry(train_X, train_Y, test_X, test_Y):
    net = RNN(1, 2, 1)
    optims = optim.Adam(net.parameters(), lr=0.1)
    loss = nn.MSELoss()
    train(net, train_X, train_Y, optims, loss)
    test(net, test_X, test_Y, loss)


if __name__ == "__main__":
    dataset = getData()
    data_X, data_Y = create_dataset(dataset)
    train_size = int(len(data_X) * TRAIN)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]
    RNNtry(train_X, train_Y, test_X, test_Y)