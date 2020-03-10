from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

#模拟文档集合
corpus = ['I like great basketball game',
          'This video game is the best action game I have ever played',
          'I really really like basketball',
          'How about this movie? Is the plot great?',
          'Do you like RPG game?',
          'You can try this FPS game',
          'The movie is really great, so great! I enjoy the plot']

#构建tfidf的值
transformer = TfidfTransformer()
vectorizer = CountVectorizer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
tfidf_array = tfidf.toarray()

#进行聚类，在我这个版本里默认使用的是欧氏距离
clusters = KMeans(n_clusters=3)
s = clusters.fit(tfidf_array)

#输出所有质心点，可以看到质心点的向量是组内成员向量的平均值
print('所有质心点的向量')
print(clusters.cluster_centers_)
print('\n')

#输出每个文档所属的分组
print('每个文档所属的分组')
print(clusters.labels_)

#输出每个分组内的文档
dict = {}
for i in range(len(clusters.labels_)):
    label = clusters.labels_[i]
    if label not in dict.keys():
        dict[label] = []
        dict[label].append(corpus[i])
    else:
        dict[label].append(corpus[i])
print(dict)