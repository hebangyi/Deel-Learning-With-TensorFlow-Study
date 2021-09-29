import tensorflow as tf

a = tf.range(16)
b = tf.reshape(a, [2, 8])
c = tf.reshape(a, [4, 4])

print(a)
print(b)
print(c)

d = tf.constant([1, 2, 3, 4])
e = tf.constant([5, 6, 7, 8])

f = tf.stack([d, e], axis=1)
print(f)
print("=====")
print(d)
print(tf.one_hot(d, depth=10))
