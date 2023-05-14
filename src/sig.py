# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 利用 numpy 实现 Sigmoid 函数
# def sigmoid(x):
#     # x 为输入值
#     # y = sigmoid(x)
#     y = 1 / (1 + np.exp(-x))
#     # dy=y*(1-y)  # 若要实现 Sigmod() 的导数图像，打开此处注释，并返回 dy 值即可。
#     return y
#
#
# # 利用 matplotlib 来进行画图
# def plot_sigmoid():
#     # param:起点，终点，间距
#     x = np.arange(-8, 8, 0.2)
#     y = sigmoid(x)
#     plt.plot(x, y)
#     plt.show()
#
#
# if __name__ == '__main__':
#     plot_sigmoid()

# coding:utf-8
import matplotlib.pyplot as plt
import os
import numpy as np


# def tanh():
# 	# 采样
#     x = np.linspace(-10,10,10000)
#     y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.set(title="tanh",xlabel="x",ylabel="y=tanh(x)")
# 	# 设置线宽和颜色
#     ax.plot(x,y,linewidth=3.0,color="red")
# 	# 保存在当前目录下
#     plt.savefig("./tanh.png",format="png")
# 	# 显示
#     plt.show()
#
# if __name__ == "__main__":
#     tanh()

import numpy as np
import matplotlib.pyplot as plt

# 0 设置字体
plt.rc('font',family='Times New Roman', size=15)

# 1.1 定义sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
# 1.2 定义tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# 1.3 定义relu函数
def relu(x):
    return np.where(x < 0, 0, x)
# 1.4 定义prelu函数
def prelu(x):
    return np.where(x<0, x * 0.5, x)

def Erelu(x):
    return np.where(x < 0, 0.01* np.exp(x)-1, x)

def lrelu(x):
    return np.where(x < 0, 0.01* x, x)

# 2.1 定义绘制函数sigmoid函数
def plot_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig = plt.figure()#如果使用plt.figure(1)表示定位（创建）第一个画板，如果没有参数默认创建一个新的画板，如果plt.figure(figsize = (2,2)) ，表示figure 的大小为宽、长
    ax = fig.add_subplot(111)#表示前面两个1表示1*1大小，最后面一个1表示第1个
    ax.spines['top'].set_color('none')#ax.spines设置坐标轴位置，set_color设置坐标轴边的颜色
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y,color="black", lw=3)#设置曲线颜色，线宽
    plt.xticks(fontsize=15)#设置坐标轴的刻度子字体大小
    plt.yticks(fontsize=15)
    plt.xlim([-10.05, 10.05])#设置坐标轴范围
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()#自动调整子图参数
    plt.show()#显示绘图
# 2.2 定义绘制函数tanh函数
def plot_tanh():
    x = np.arange(-10, 10, 0.1)
    y = tanh(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-10.05, 10.05])
    plt.ylim([-0.02, 1.02])
    ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
    ax.set_xticks([-10, -5, 5, 10])
    plt.tight_layout()
    plt.show()
# 2.3 定义绘制函数relu函数
def plot_relu():
    x = np.arange(-10, 10, 0.1)
    y = relu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-10.05, 10.05])
    plt.ylim([-0.02, 1.02])
    ax.set_yticks([2, 4, 6, 8, 10])
    plt.tight_layout()
    plt.show()


def plot_lrelu():
    x = np.arange(-10, 10, 0.1)
    y = lrelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xlim([-10.05, 10.05])
    # plt.ylim([-0.02, 1.02])
    # ax.set_yticks([2, 4, 6, 8, 10])
    plt.tight_layout()
    plt.show()

def plot_Erelu():
    x = np.arange(-10, 10, 0.1)
    y = Erelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xlim([-10.05, 10.05])
    # plt.ylim([-0.02, 1.02])
    # ax.set_yticks([2, 4, 6, 8, 10])
    plt.tight_layout()
    plt.show()


# 2.4 定义绘制函数prelu函数
def plot_prelu():
    x = np.arange(-10, 10, 0.1)
    y = prelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()


# 3 运行程序
# plot_sigmoid()
plot_tanh()
# plot_relu()
# plot_lrelu()
# plot_Erelu()
