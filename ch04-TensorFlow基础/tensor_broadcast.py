import tensorflow as tf

# 小维度从右方对齐,其他的维度可以没有或者为1
a = tf.random.normal([4, 32, 32, 3])
b = tf.random.normal([3])
print((a + b).shape)  # + 默认调用了 broadcast

c = tf.random.normal([32, 32, 1])  # 默认最后的一个维度扩展为3的复制
print((a + c).shape)

# 将低维度的转换为特定的维度
print(tf.broadcast_to(tf.random.normal([4, 1, 1, 1]), [4, 32, 32, 3]))
