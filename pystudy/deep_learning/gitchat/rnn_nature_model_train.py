#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : rnn_nature_model_train.py
@Author : jeffsheng
@Date : 2020/2/6 0006
@Desc : 
"""

import numpy as np

print("---------step1:>>>创建两个字典---------------------------")
with open('./temp/pg2265.txt', 'r', encoding='utf-8') as f:
    text=f.read()

# 移除文本前面的描述部分
text = text[15858:]

# chars：是输入文本中无重复字符的集合
chars = set(text)
"""
char2int：用语料库中所有文本的无重复字符建立的字典，例如，
“e”:0,“h”:1,“l”:2,“o”:3，用来将字符和整数对应起来，可以将一个文本转化为整数数组。
"""
char2int = {ch:i for i,ch in enumerate(chars)}
# int2char：将整数对应到字符，这个字典在后面会被用来将模型的输出转化成字符，进而得到文本
int2char = dict(enumerate(chars))
# text_ints：根据词汇字典，将整个文本转化为整数序列
text_ints = np.array([char2int[ch] for ch in text], dtype=np.int32)


print("---------step2:>>> 生成 x 和 y 的批次序列数据---------------------------")


def reshape_data(sequence, batch_size, num_steps):
    """
    目标是根据目前为止观察到的字符序列来预测下一个字符， 因此，我们要将网络的输出和输入之间错开一个字符
    :param sequence: 语料库中的字符所对应的整数数据
    :param batch_size:训练集的行数
    :param num_steps:rnn的cell时间步数
    :return:
    """
    mini_batch_length = batch_size * num_steps
    num_batches = int( len(sequence) / mini_batch_length )

    # 序列尾部不满 mini_batch_length 的部分就忽略了
    if num_batches * mini_batch_length + 1 > len(sequence):
        num_batches = num_batches - 1

    # y 和 x 之间错开一位
    x = sequence[0 : num_batches * mini_batch_length]
    y = sequence[1 : num_batches * mini_batch_length + 1]

    # 将 x 和 y 分成批次数据，分成 batch_size 批
    x_batch_splits = np.split(x, batch_size)
    y_batch_splits = np.split(y, batch_size)

    # 将批次数据堆叠起来，行数 = batch_size，列数 = num_steps * num_batches
    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)

    return x, y


print("-----------将 x 和 y 划分为 mini-batches-----------")
np.random.seed(123)


# 定义 create_batch_generator 将列数截成 num_batches 段，每段有 num_steps 列
def create_batch_generator(data_x, data_y, num_steps):
    # data_x 的行数等于 batch_size，列数等于 num_steps * num_batches
    batch_size, tot_batch_length = data_x.shape
    num_batches = int( tot_batch_length / num_steps )

    # 将列数截成 num_batches 段，每段有 num_steps 列
    for b in range(num_batches):
        yield (data_x[:, b * num_steps: (b + 1) * num_steps],
               data_y[:, b * num_steps: (b + 1) * num_steps])


print("--------------step3:>>>建立 LSTM 模型------------------")
import tensorflow as tf
import os



class CharRNN(object):

    """
    sampling:在训练和采样时会用两个不同的计算图，所以需要给 constructor 加一个 Boolean 型参数来决定我们是要训练模型，还是要采样。
    其中参数 sampling 为 false 时是为了训练，为 true 时是采样
    """
    def __init__(self, num_classes, batch_size=64,
                 num_steps=100, lstm_size=128,
                 num_layers=1, learning_rate=0.001,
                 keep_prob=0.5, grad_clip=5,
                 sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        # 参数 grad_clip 是用来做梯度剪切的，以应对梯度爆炸问题
        self.grad_clip = grad_clip

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build(sampling=sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()


    def build(self, sampling):
        # 采样模式下一个字符一个字符生成文本
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps

        # 建立三个 placeholder： tf_x, tf_y, tf_keepprob ，用来喂入输入数据
        tf_x = tf.placeholder(tf.int32,
                              shape=[batch_size, num_steps],
                              name='tf_x')
        tf_y = tf.placeholder(tf.int32,
                              shape=[batch_size, num_steps],
                              name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32,
                                     name='tf_keepprob')

        # 1. One-hot 编码
        ## depth：定义one-hot 向量的维度是 num_classes x 1，在每个字符所对应的索引位置为 1，其余位置为 0
        # num_classes 是语料库中无重复字符的总数
        x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth=self.num_classes)

        # 2. 定义多层 LSTM cell
        ## 从内向外依次为，用 BasicLSTMCell 建立 cell，
        ## 用 DropoutWrapper 应用 dropout
        ## 用 MultiRNNCell 建立多层 LSTM
        cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
                output_keep_prob=tf_keepprob)
                for _ in range(self.num_layers)])

        # 3. 定义 cell 的初始状态
        ## 回忆一下 LSTM 的结构，在每个时间步，需要用到之前一步的cell状态，
        ## 所以当开始处理一个新的序列时，首先将 cell 的状态初始化为 0，
        self.initial_state = cells.zero_state(
            batch_size, tf.float32)

        # 4. 建立具有前两步定义的 cell 和 初始状态的 RNN，
        ## 用输入序列数据，LSTM cell，初始状态，生成 LSTM 的展开后的结构
        ## 输出 lstm_outputs 的维度是 (batch_size, num_steps, lstm_size)
        ## self.final_state 会被存起来用作下一个批次数据的初始状态
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            cells, x_onehot,
            initial_state=self.initial_state)

        print('  << lstm_outputs  >>', lstm_outputs)

        # 5. 将形状变为 [batch_size * num_steps, lstm_size]
        seq_output_reshaped = tf.reshape(
            lstm_outputs,
            shape=[-1, self.lstm_size],
            name='seq_output_reshaped')

        # 6. 经过全连接层，得到 logits
        logits = tf.layers.dense(
            inputs=seq_output_reshaped,
            units=self.num_classes,
            activation=None,
            name='logits')

        # 7. 得到下一批字符的概率
        ## 将 output 传递给 softmax 层，用来得到概率，
        ## 这样输出向量的每个值都在 0～1 之间，并且和为 1
        ## softmax 层和输出层有相同的维度，vocab_size x 1
        ## `$y^t[i]$` 表示在时刻 t 时，索引 i 对应的字符为预测得到的下一个字符的概率
        proba = tf.nn.softmax(
            logits,
            name='probabilities')

        print(proba)

        y_reshaped = tf.reshape(
            y_onehot,
            shape=[-1, self.num_classes],
            name='y_reshaped')

        # 8. 定义损失函数
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=y_reshaped),
            name='cost')

        # 9. 梯度剪切避免梯度爆炸问题
        ## LSTM 虽然对基础对 RNN 的梯度消失问题有所改进，但还是存在梯度爆炸问题
        ## 所以这里我们用 gradient clipping 技术来应对此问题，
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars),
            self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            name='train_op')


    def train(self, train_x, train_y,
              num_epochs, ckpt_dir='./model/'):

        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)

            n_batches = int(train_x.shape[1] / self.num_steps)
            iterations = n_batches * num_epochs

            for epoch in range(num_epochs):

                # 在每个 epoch 开始时，RNN cell 以零初始状态开始
                new_state = sess.run(self.initial_state)
                loss = 0

                ## 生成 Mini batch 数据
                bgen = create_batch_generator(
                    train_x, train_y, self.num_steps)

                for b, (batch_x, batch_y) in enumerate(bgen, 1):
                    iteration = epoch * n_batches + b

                    # 喂入数据 batch_x, batch_y，训练每一小批数据
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keepprob:0': self.keep_prob,
                            self.initial_state: new_state}

                    # 当执行完每一小批数据时，将状态更新为 dynamic_rnn 返回的 final_state
                    # 这个更新后的状态会被用来执行下一小批数据
                    # 当前状态会随着迭代的次数不断更新
                    batch_cost, _, new_state = sess.run(
                        ['cost:0', 'train_op',
                         self.final_state],
                        feed_dict=feed)

                    if iteration % 10 == 0:
                        print('Epoch %d/%d Iteration %d'
                              '| Training loss: %.4f' % (
                                  epoch + 1, num_epochs,
                                  iteration, batch_cost))

                ## 保存训练好的模型
                self.saver.save(
                    sess, os.path.join(
                        ckpt_dir, 'language_modeling.ckpt'))


    def get_top_char(probas, char_size, top_n=5):
        p = np.squeeze(probas)
        p[np.argsort(p)[:-top_n]] = 0.0
        p = p / np.sum(p)
        ch_id = np.random.choice(char_size, 1, p=p)[0]
        return ch_id




    def sample(self, output_length,
               ckpt_dir, starter_seq="The "):

        # 给一个初始序列作为生成文本的开头，生成的序列从参数 starter_seq 开始
        observed_seq = [ch for ch in starter_seq]

        # 调用训练好的模型
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(
                sess,
                tf.train.latest_checkpoint(ckpt_dir))

            new_state = sess.run(self.initial_state)

            # 将 starter sequence 输入到 LSTM 模型中
            for ch in starter_seq:
                x = np.zeros((1, 1))
                x[0, 0] = char2int[ch]

                feed = {'tf_x:0': x,
                        'tf_keepprob:0': 1.0,
                        self.initial_state: new_state}

                # self.final_state 是 tf.nn.dynamic_rnn 的输出
                proba, new_state = sess.run(
                    ['probabilities:0', self.final_state],
                    feed_dict=feed)

            # 随机采样了一个 index
            ch_id = get_top_char(proba, len(chars))

            # 将这个 index 转化为字符，加入到要生成的序列中
            observed_seq.append(int2char[ch_id])

            # 将每次采样新得到的 index 向量输入到 LSTM 模型中，
            ## 得到下一个概率，选择下一个 index，将对应的字符附加到序列上，一直到达到了要求的长度
            for i in range(output_length):
                x[0, 0] = ch_id

                feed = {'tf_x:0': x,
                        'tf_keepprob:0': 1.0,
                        self.initial_state: new_state}

                proba, new_state = sess.run(
                    ['probabilities:0', self.final_state],
                    feed_dict=feed)

                ch_id = get_top_char(proba, len(chars))

                observed_seq.append(int2char[ch_id])

        return ''.join(observed_seq)


print("-----------------训练模型--------------------------")
batch_size = 64
num_steps = 100

train_x, train_y = reshape_data(text_ints,
                                batch_size,
                                num_steps)

rnn = CharRNN(num_classes = len(chars), batch_size = batch_size)

rnn.train(train_x, train_y,
          num_epochs = 100,
          ckpt_dir='./model-100/')


print("---------------采样生成文本---------------------------")
np.random.seed(123)

rnn = CharRNN(len(chars), sampling = True)

print(rnn.sample(ckpt_dir = './model-100/',
                 output_length = 500))