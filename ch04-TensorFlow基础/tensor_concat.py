import tensorflow as tf

a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)

print(c.shape)

d = tf.ones([4, 35, 8])
e = tf.ones([4, 3, 8])
# 通过维度进行合并
# 在第二个维度concat 两个shape除了concat维度,其他的维度相同
print(tf.concat([d, e], axis=1).shape)

f = tf.ones([4, 35, 8])
g = tf.ones([4, 35, 8])
# 在0这个维度上合并两个维度,在0维度上生成一个新的维度,stack两个shape必须相等
print(tf.stack([f, g], axis=0).shape)

h = tf.ones([2, 4, 35, 8])
aa, bb = tf.unstack(h, axis=0)
print(aa.shape)
print(bb.shape)
a1, a2, a3, a4 = tf.unstack(h, axis=1)
print(a1.shape)
print(a2.shape)
print(a3.shape)
print(a4.shape)
