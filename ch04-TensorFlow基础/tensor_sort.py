import tensorflow as tf

# tensorflow 对象排序
a = tf.random.shuffle(tf.range(5))
b = tf.sort(a, direction='DESCENDING')
c = tf.argsort(a, direction='DESCENDING')
print(b)  # 将值排序后返回
print(c)  # 返回最大值和次大值的索引顺序

d = tf.random.normal([5, 5])
print(d)
e = tf.math.top_k(d, 2)  # 取出最后两个维度中的最大的两个值的索引
print(e)
print("====")
print(e.indices)  # 返回索引值
print(e.values)  # 返回值

# 使用top_k的应用
prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
target = tf.constant([2, 0])

k_b = tf.math.top_k(prob, 3).indices
print("shape k_b = ", k_b.shape)
k_b = tf.transpose(k_b, [1, 0])
target = tf.constant([2, 0])

print(k_b)
print(target)
