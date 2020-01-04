# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from pystudy.common.optimizer import *

class Trainer:
    """进行神经网络的训练的类
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        # 根据mini_batch_size求出每个epoch需要迭代训练数据的次数
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 求出epochs*iter_per_epoch的总的迭代次数
        self.max_iter = int(epochs * self.iter_per_epoch)
        # 当前第几次迭代
        self.current_iter = 0
        # 当前第几个epoch
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 随机抽取batch_size个样本下标
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        # 根据下标获取样本子集
        x_batch = self.x_train[batch_mask]
        # 根据下标获取标签子集
        t_batch = self.t_train[batch_mask]
        # 反向传播求梯度
        grads = self.network.gradient(x_batch, t_batch)
        # 根据优化算法更新梯度
        self.optimizer.update(self.network.params, grads)
        # 更新完梯度重新求损失函数
        loss = self.network.loss(x_batch, t_batch)
        # 记录本次迭代训练的损失值
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))
        # 当前迭代次序达到每个epoch的迭代次数时，current_epoch+1
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            # 抽样预测（计算）训练和测试数据的准确率
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        # 迭代总的迭代次数
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)
        # 打印准确率
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

