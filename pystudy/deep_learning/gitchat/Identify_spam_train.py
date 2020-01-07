#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : Identify_spam_train.py
@Author : jeffsheng
@Date : 2020/1/7
@Desc : rnn识别垃圾邮件应用
"""
# 加载需要的包
import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
from pystudy.config.log_config import log_base_config
logger = log_base_config.get_log("intent_train")
# 将 default graph 重新初始化，保证内存中没有其他的 Graph，相当于清空所有的张量
ops.reset_default_graph()

# 为 graph 建立会话 session
sess = tf.Session()


# 设定 RNN 模型的参数
epochs = 30
batch_size = 250
# 每个文本的最大长度为 25 个单词，这样会将较长的文本剪切为 25 个，不足的用零填充
max_sequence_length = 25
rnn_size = 10
# 每个单词都将被嵌入到一个长度为 50 的可训练向量中
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

output_path = './model/classifier_save/normal/'
model_output_path = os.path.join(output_path, "model.ckpt")

# 导入数据
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 直接下载zip格式的数据集
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    # 格式化数据
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')

    # 将数据存储到 text 文件
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    # 从 text 文件打开数据
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]


# 数据预处理
# 首先，将数据中的标签和邮件文本分开，得到 text_data_target 和 text_data_train
text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# 为了减少 vocabulary, 先清理文本，移除特殊字符，删掉多余的空格，将文本都换成小写
# 创建一个文本清理函数
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


# 调用 clean_text 清理文本
text_data_train = [clean_text(x) for x in text_data_train]


# 接下来将文本转为词的 ID 序列
# 用 TensorFlow 中一个内置的 VocabularyProcessor 函数来处理文本
"""
tf.contrib.learn.preprocessing.VocabularyProcessor (max_document_length, min_frequency=0, vocabulary=None, tokenizer_fn=None)
- max_document_length: 是文本的最大长度。如果文本的长度大于这个值，就会被剪切，小于这个值的地方用 0 填充。
- min_frequency: 是词频的最小值。当单词的出现次数小于这个词频，就不会被收录到词表中。
- vocabulary: CategoricalVocabulary 对象。
- tokenizer_fn：分词函数

使用这个函数时一般分为几个动作：
1.首先将列表里面的词生成一个词典；
2.按列表中的顺序给每一个词进行排序，每一个词都对应一个序号(从1开始，<UNK>的序号为0)
3.按照原始列表顺序，将原来的词全部替换为它所对应的序号
4.同时如果大于最大长度的词将进行剪切，小于最大长度的词将进行填充
5.然后将其转换为列表，进而转换为一个array

"""
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))


# shuffle，可以打乱数据行序，使数据随机化
text_processed = np.array(text_processed)
# 将text_data_target转为0和1的结果标签
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
'''
shuffle与permutation的区别
    1函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    2区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    3而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
'''
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
# 按照随机打乱词向量及标签的顺序重新返回数据集和训练集
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

'''
shuffle 数据后，将数据集分为 80% 训练集和 20% 测试集 
如果想做交叉验证 cross-validation ，可以将 测试集 进一步分为测试集和验证集来调参
'''
ix_cutoff = int(len(y_shuffled)*0.80)
# 分割数据：训练集和测试集
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
# 分割标签：训练集标签和测试集标签
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
# 词向量的大小
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# 将输入数据做词嵌入 Embedding
# 首先，建立 x 和 y 的 placeholders
# x_data 的大小为 [None, maxsequencelength]，maxsequencelength就是每个文本句子的组成单词数量
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
# y_output 是一个整数，值为 0 或 1, 分别表示 ham 和 spam
y_output = tf.placeholder(tf.int32, [None])

# 接下来，建立 embedding
# 用 embedding 将索引转化为特征，这是一种特征学习方法，可以用来自动学习数据集中各个单词的显著特征,
# 它可以将单词映射到固定长度为 embedding_size 大小的向量
"""
建立嵌入层的步骤：
1 首先建立一个嵌入矩阵 embedding_mat，它是一个 tensor 变量，大小为 [vocab_size × embedding_size]，
将它的元素随机初始化为 [-1, 1] 之间
    解释：vocab_size表示文本中所有不重复单词总数，embedding_size表示每个单词的向量长度
2 然后用 tf.nn.embedding_lookup 函数来找嵌入矩阵 embedding_mat 中与 x_data 的每个元素所对应的行，
也就是将每个单词的整数索引，映射到这个可训练的嵌入矩阵 embedding_mat 的某一行
"""
# 初始化词嵌入矩阵
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

# 建立RNN模型
# 定义 RNN cell
"""
BasicRNNCell 是 RNN 的基础类， 激活函数默认是 tanh， 每调用一次，就相当于在时间上“推进了一步”， 
它的参数 num_units 就是隐藏层神经元的个数

RNNCell解释：
    BasicRNNCell 和 BasicLSTMCell 都是抽象类 RNNCell 的两个子类， 每个 RNNCell 都有一个 call 方法，
使用为 :   (output, next_state) = call(input, state)
其中输入数据input的形状为 (batch_size, input_size)， 得到的隐层状态形状 (batch_size, state_size)，输出形状为 (batch_size, output_size)
比如：输入一个初始状态 h0 和输入 x1，调用 call(x1, h0) 后就可以得到 (output1, h1)， 再调用一次 call(x2, h1) 就可以得到 (output2, h2)


"""
print("-------------------建立RNN模型-----------------------------")
if tf.__version__[0] >= '1':
    cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)

# 用 tf.nn.dynamic_rnn 建立 RNN 序列
"""
tf.nn.dynamic_rnn:
1 单个的 RNNCell，调用一次就只是在序列时间上前进了一步。 
  所以需要使用 tf.nn.dynamic_rnn 函数，它相当于调用了n次 RNNCell,即通过 {h0,x1, x2, …., xn} 直接得到 {h1,h2…,hn},{output1,output2…,outputn}
2 tf.nn.dynamic_rnn 以前面定义的 cell 为基本单元建立一个展开的RNN序列网络,将词嵌入和初始状态输入其中,返回了 output，还有最后的 state。
3 output是一个三维的tensor，是time_steps步的所有的输出，形状为 (batch_size, time_steps, cell.output_size)
   state是最后一步的隐状态，形状为 (batch_size, cell.state_size)
"""
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)

# 再为 RNN 添加 dropout
'''
tf.nn.dropout 用来减轻过拟合
dropout_keep_prob 是保留比例，是神经元被选中的概率，和输入一样，也是一个占位符，取值为 (0,1] 之间
'''
output = tf.nn.dropout(output, dropout_keep_prob)
'''
tf.transpose 用于将张量进行转置，张量的原有维度顺序是 [0, 1, 2], 则 [1, 0, 2] 是告诉 tf 要将 0 和 1 维转置
 0 代表三维数组的高，1 代表二维数组的行，2 代表二维数组的列。 
 即将输出 output 的维度 [batch_size, time_steps, cell.output_size] 变成 [time_steps, batch_size, cell.output_size]
'''
output = tf.transpose(output, [1, 0, 2])
# 切掉最后一个时间步的输出作为预测值 tf.gather 用于将向量中某些索引值提取出来，得到新的向量。
last = tf.gather(output, int(output.get_shape()[0]) - 1)

# 将 output 传递给一个全连接层，来得到 logits_out
'''
为了完成 RNN 的分类预测，通过一个全连接层 将 rnn_size 长度的输出变为二分类输出
在 RNN 中，全连接层可以将 embedding 空间拉到隐层空间，将隐层空间转回 label 空间

tf.truncated_normal函数：
    用来从截断的正态分布中随机抽取值，即生成的值服从指定平均值和标准偏差的正态分布， 
但是如果生成的值与均值的差值大于两倍的标准差，即在区间（μ-2σ，μ+2σ）之外，
则丢弃并重新进行选择，这样可以保证生成的值都在均值附近
其中参数 shape 表示生成的张量的维度是 [rnn_size, 2]， mean 均值默认为 0， stddev 标准差设置为 0.1。
'''
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
# logits 是这个全连接层的输出，作为 softmax 的输入，在接下来定义损失函数时用到
logits_out = tf.matmul(last, weight) + bias


# 定义损失函数和准确率函数
'''
tf.nn.sparse_softmax_cross_entropy_with_logits:
sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
logits：是神经网络最后一层的输出，如果有 batch 的话，形状是 [batch_size, num_classes]
labels：是实际的标签，形状是 [batch_size, 1],每个 label 的取值是 [0, num_classes) 的离散值，是哪一类就标记哪个类对应的 label

    这个函数将 softmax 和 cross_entropy 放在一起计算， 先对网络最后一层的输出做一个 softmax 求取输出属于某一类的概率， 
然后 softmax 的输出向量再和样本的实际标签做一个交叉熵 cross_entropy，来计算的两个概率分布之间的距离，这是分类问题中使用比较广泛的损失函数之一
'''
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)

# 上面损失函数的返回值是一个向量，是一个 batch 中每个样本的 loss，不是一个数， 需要通过 tf.reduce_mean 对向量求均值，计算 batch 内的平均 loss。
loss = tf.reduce_mean(losses)

# 定义准确率函数
'''
tf.argmax 用来返回最大值 1 所在的索引位置，因为标签向量是由 0,1 组成，因此返回了预测类别标签， 
再用 tf.equal 来检测预测与真实标签是否匹配，返回一个布尔数组,
用 tf.cast 将布尔值转换为浮点数 [1,0,1,1...] 最后用 tf.reduce_mean 计算出平均值即为准确率
'''
accuracy = tf.reduce_mean(tf.cast( tf.equal( tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64) ), tf.float32 ))

# 选择优化算法
'''
RMSProp 是 Geoff Hinton 提出的一种自适应学习率方法，为了解决 Adagrad 学习率急剧下降问题的,
这种方法很好的解决了深度学习中过早结束的问题，适合处理非平稳目标，对于RNN效果很好
'''
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

print("--------------训练模型-------------")
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
'''
训练的流程为：
1 Shuffle 训练数据：在每一代 epoch 遍历时都 shuffle 一下数据可以避免过度训练
2 选择训练集, 训练每个 batch
3 计算训练集和测试集的损失 loss 和准确率 accuracy
'''
# 开始训练
for epoch in range(epochs):

    # Shuffle 训练集
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    # 用 Mini batch 梯度下降法
    num_batches = int(len(x_train)/batch_size) + 1

    for i in range(num_batches):
        # 选择每个 batch 的训练数据
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]

        # 进行训练： 用 Session 来 run 每个 batch 的训练数据，逐步提升网络的预测准确性
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)

    # 将训练集每一代的 loss 和 accuracy 加到整体的损失和准确率中去
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # 同时计算并记录测试集每一代的损失和准确率
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))

# 将模型保存到save/model.ckpt文件
saver_path = saver.save(sess, model_output_path)
logger.info("Final Model saved in file: %s" % saver_path)

print("------------可视化损失函数 loss 和准确率 accuracy-----------")
# 损失随着时间的变化
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# 准确率随着时间的变化
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

