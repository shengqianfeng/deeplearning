import tensorflow as tf
import os
# 屏蔽不支持avx2指令警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义常量操作 hello
hello = tf.constant("hello tensorflow")
# 创建一个会话
session = tf.compat.v1.Session()
print(session.run(hello))

