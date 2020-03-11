#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : cbow_word2vec.py
@Author : jeffsheng
@Date : 2020/3/10
@Desc : 用 TensorFlow 建立 CBOW 模型训练词向量
"""
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk                             # 用来做预处理
nltk.download('punkt')
import operator                            # 用来对字典里的词进行排序
from math import ceil
import csv


# 下载 Wikipedia 数据
url = 'http://www.evanjones.ca/software/'

def maybe_download(filename, expected_bytes):
  if not os.path.exists(filename):
    print('Downloading file...')
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035)


# 读取数据，并用 NLTK 进行预处理
def read_data(filename):
  with bz2.BZ2File(filename) as f:

    data = []
    file_size = os.stat(filename).st_size
    chunk_size = 1024 * 1024                                         # 每次读取 1 MB 的数据
    print('Reading data...')
    for i in range(ceil(file_size//chunk_size)+1):
        bytes_to_read = min(chunk_size,file_size-(i*chunk_size))
        file_string = f.read(bytes_to_read).decode('utf-8')
        file_string = file_string.lower()                            # 转化为小写字母
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
  return data

words = read_data('wikipedia2text-extracted.txt.bz2')
# 打印单词数量
print('Data size %d' % len(words))
print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])


# 将词汇表大小限制在 50000
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  # 选择 50000 个最常见的单词，其他的用 UNK 代替
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  # 为每个 word 建立一个 ID，保存到一个字典里
  for word, _ in count:
    dictionary[word] = len(dictionary)

  data = list()
  unk_count = 0


  # 将文本数据中所有单词都换成相应的 ID，得到 data
  for word in words:
      # 不在字典里的单词，用 UNK 代替
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)

  count[0][1] = unk_count

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  # 确保字典和词汇表的大小一样
  assert len(dictionary) == vocabulary_size

  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words

data_index = 0



def generate_batch_cbow(batch_size, window_size):

    global data_index
    # 每次读取一个数据集后增加 1
    # span 是指中心词及其两侧窗口的总长度
    span = 2 * window_size + 1
    # batch 为上下文数据集，一共有 span - 1 个列
    batch = np.ndarray(shape=(batch_size,span-1), dtype=np.int32)
    # labels 为中心词
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # buffer 用来存储 span 内部的数据
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 批次读取数据，对于每一个 batch index 遍历 span 的所有元素，填充到 batch 的各个列中
    for i in range(batch_size):
        # 在 buffer 的中心是中心词
        target = window_size
        # 因为只需要考虑中心词的上下文，这时不需要考虑中心词
        target_to_avoid = [ window_size ]

        # 将选定的中心词加入到 avoid_list 下次时用
        col_idx = 0
        for j in range(span):
            # 建立批次数据时忽略中心词
            if j==span//2:
                continue
            batch[i,col_idx] = buffer[j]
            col_idx += 1
        labels[i, 0] = buffer[target]

        # 每次读取一个数据点，需要移动 span 一次，建立一个新的 span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

for window_size in [1,2]:
    data_index = 0
    batch, labels = generate_batch_cbow(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' % (window_size))
    print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])



# 设置超参数
batch_size = 128
# embedding 向量的维度
embedding_size = 128
# 中心词的左右两边各取 2 个
window_size = 2
# 随机选择一个 validation 集来评估单词的相似性
valid_size = 16
# 从一个大窗口随机采样 valid 数据
valid_window = 50

# 选择 valid 样本时, 取一些频率比较大的单词，也选择一些适度罕见的单词
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)
# 负采样的样本个数
num_sampled = 32


# 定义输入输出
tf.reset_default_graph()
# 用于训练的上下文数据，有 2*window_size 列
train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*window_size])
# 用于训练的中心词
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# Validation 不需要用 placeholder，因为前面已经定义了 valid_examples
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 定义模型参数
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0,dtype=tf.float32))

# Softmax 的权重和 Biases
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                 stddev=0.5 / math.sqrt(embedding_size),dtype=tf.float32))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))


# 为 input 的每一列做 embedding lookups
# 然后求平均，得到大小为 embedding_size 的词向量
stacked_embedings = None
print('Defining %d embedding lookups representing each word in the context'%(2*window_size))
for i in range(2*window_size):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
    x_size,y_size = embedding_i.get_shape().as_list()
    if stacked_embedings is None:
        stacked_embedings = tf.reshape(embedding_i,[x_size,y_size,1])
    else:
        stacked_embedings = tf.concat(axis=2,values=[stacked_embedings,tf.reshape(embedding_i,[x_size,y_size,1])])

assert stacked_embedings.get_shape().as_list()[2]==2*window_size
print("Stacked embedding size: %s"%stacked_embedings.get_shape().as_list())
mean_embeddings =  tf.reduce_mean(stacked_embedings,2,keepdims=False)
print("Reduced mean embedding size: %s"%mean_embeddings.get_shape().as_list())

# 每次用一些负采样样本计算 softmax loss,
# 输入是训练数据的 embeddings
# 用这个 loss 来优化 weights, biases, embeddings
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))


# 定义优化算法
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# 用余弦距离来计算样本的相似性
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# 训练cbow模型
num_steps = 100001
cbow_losses = []

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:        # ConfigProto 可以提供不同的配置设置

    tf.global_variables_initializer().run()                                            # 初始化变量
    print('Initialized')

    average_loss = 0

    # 训练 Word2vec 模型 num_step 次
    for step in range(num_steps):

        # 生成一批数据
        batch_data, batch_labels = generate_batch_cbow(batch_size, window_size)

        # 运行优化算法，计算损失
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # 更新平均损失变量
        average_loss += l

        if (step+1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
                # 计算一下平均损失
            cbow_losses.append(average_loss)
            print('Average loss at step %d: %f' % (step+1, average_loss))
            average_loss = 0

        # 评估 validation 集的单词相似性
        if (step+1) % 10000 == 0:
            sim = similarity.eval()
            # 对验证集的每个用于验证的中心词，计算其 top_k 个最近单词的余弦距离，
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8                                                             # 近邻的数目
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    cbow_final_embeddings = normalized_embeddings.eval()


np.save('cbow_embeddings',cbow_final_embeddings)

with open('cbow_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(cbow_losses)


# 可视化 embedding 的结果
def find_clustered_embeddings(embeddings,distance_threshold,sample_threshold):

    # 计算余弦距离
    cosine_sim = np.dot(embeddings,np.transpose(embeddings))
    norm = np.dot(np.sum(embeddings**2,axis=1).reshape(-1,1),np.sum(np.transpose(embeddings)**2,axis=0).reshape(1,-1))
    assert cosine_sim.shape == norm.shape
    cosine_sim /= norm

    np.fill_diagonal(cosine_sim, -1.0)

    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    # 找到每次循环的最大值，来计数是否超过阈值
    for _ in range(sample_threshold-1):
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]),argmax_cos_sim] = -1

    max_cosine_sim = np.max(mod_cos_sim,axis=1)

    return np.where(max_cosine_sim>distance_threshold)[0]


num_points = 1000 # 用一个大的样本空间来构建 T-SNE，然后用余弦相似性对其进行修剪

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('Fitting embeddings to T-SNE. This can take some time ...')

selected_embeddings = cbow_final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')

# 修剪词嵌入，只取超过相似性阈值的 n 个样本，使可视化变得整齐一些

selected_ids = find_clustered_embeddings(selected_embeddings,.25,10)
two_d_embeddings = two_d_embeddings[selected_ids,:]

print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')


def plot(embeddings, labels):

  n_clusters = 20                                                                     # clusters 的数量

  label_colors = [pylab.cm.Spectral(float(i) /n_clusters) for i in range(n_clusters)]        # 为每个 cluster 自动分配颜色

  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

  # 定义 K-Means
  kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
  kmeans_labels = kmeans.labels_

  pylab.figure(figsize=(15,15))

  # 画出所有 embeddings 和他们相应的单词
  for i, (label,klabel) in enumerate(zip(labels,kmeans_labels)):
    x, y = embeddings[i,:]
    pylab.scatter(x, y, c=label_colors[klabel])

    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom',fontsize=10)

  pylab.show()

words = [reverse_dictionary[i] for i in selected_ids]
plot(two_d_embeddings, words)