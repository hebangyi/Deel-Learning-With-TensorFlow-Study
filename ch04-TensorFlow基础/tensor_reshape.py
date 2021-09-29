import tensorflow as tf

# tensor的维度变换
a = tf.random.normal([4, 28, 28, 3])
print(a.shape, a.ndim)
print(tf.reshape(a, [4, 28 * 28, 3]).shape)  # 变换的方式只要数据的个数保持一致
print(tf.reshape(a, [4, -1, 3]).shape)  # 变换的方式只要数据的个数保持一致, -1表示自动计算,只能有一个-1
print(tf.reshape(a, [4, 784 * 3]).shape)
print(tf.reshape(a, [4, -1]).shape)

# 维度转置
b = tf.random.normal([4, 3, 2, 1])
print(b.shape)
print(tf.transpose(b).shape)

# 维度的自由交换
print(tf.transpose(b, perm=[0, 1, 3, 2]).shape)

# 增加一个维度和减少一个维度,axis 表示增加一个维度的位置
c = tf.random.normal([4, 35, 8])
print(tf.expand_dims(a, axis=0).shape)
print(tf.expand_dims(a, axis=3).shape)
print(tf.expand_dims(a, axis=-1).shape)  # 在最后一个位置增加一个维度

# 减少一个维度 只能减少axis的值是1的数据
d = tf.random.normal([1, 2, 1, 1, 3])
print(d.shape)
print(tf.squeeze(d, axis=0).shape)
print(tf.squeeze(d, axis=-2).shape)

# 矩阵的扩张
a = tf.reshape(tf.range(9), [3, 3])
print(a.shape)
print(tf.pad(a, [[0, 0], [0, 0]]).shape)  # 因为shape 只有两个维度,所以参数只有两个数组值
print(tf.pad(a, [[1, 0], [0, 0]]))  # 行padding
print(tf.pad(a, [[0, 1], [0, 0]]))

print(tf.pad(a, [[0, 0], [1, 0]]))  # 列padding
print(tf.pad(a, [[0, 0], [0, 1]]))  # 列padding
# 在图片或句子中为了等长,进行定长处理
print(tf.pad(a, [[0, 0], [1, 1]], constant_values=5))  # 列padding

b = tf.random.normal([4, 28, 28, 3])
print(b.shape)
c = tf.pad(b, [[0, 0], [2, 2], [2, 2], [0, 0]])
print(c.shape)


# tile 矩阵的复制
a = tf.reshape(tf.range(9), [3, 3])
print(a)
print(tf.tile(a, [1, 2]))  # 分别表示某个维度复制的次数
print(tf.tile(a, [2, 1]))

