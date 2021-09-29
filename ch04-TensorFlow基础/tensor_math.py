import tensorflow as tf

a = tf.fill([2, 2], 2.)
b = tf.ones([2, 2])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)  # 整除
print(a % b)  # 余除

print(tf.math.log(a))
print(tf.exp(a))  # 指数函数

print(a ** 2)  # a 的平方
print(tf.pow(a, 3))  # a 的三次方
print(tf.sqrt(a))  # a 开方

print(tf.matmul(a, b))  # 矩阵相乘
print(a @ b)
# 如果高于两个维度,最低的两个维度相乘

