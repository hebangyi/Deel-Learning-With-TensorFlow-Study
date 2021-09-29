import numpy as np
import tensorflow as tf

# tensorflow 的对象操作
# Create Constant
print(tf.constant(1))
print(tf.constant(1.))
# print(tf.constant(2.2, dtype=tf.int32))
# TypeError: Cannot convert 2.2 to EagerTensor of dtype int32

print(tf.constant(2., dtype=tf.double))

print(tf.constant('hello,world'))

# Tensor Property
# 定义一个cpu的常量
with tf.device("cpu"):
    a = tf.constant([1])  # 在cpu上初始化a常量

with tf.device("gpu"):
    b = tf.range(4)  # 在gpu上初始化b常量

# 常量设备信息
print(a.device)  # /job:localhost/replica:0/task:0/device:CPU:0
print(b.device)  # /job:localhost/replica:0/task:0/device:GPU:0

# 将常量转换设备
aa = a.gpu()
print(aa.device)  # /job:localhost/replica:0/task:0/device:GPU:0
bb = b.cpu()
print(bb.device)  # /job:localhost/replica:0/task:0/device:CPU:0

# 将数据类型转换为numpy对象返回
print(b.numpy())  # [0 1 2 3]

# 返回数据维度
print(b.ndim)  # 1

# 描述数据信息,
print(tf.rank(b))  # tf.Tensor(1, shape=(), dtype=int32)
print(tf.rank(tf.ones([3, 4, 2])))  # tf.Tensor(3, shape=(), dtype=int32)

# tensor 类型
a1 = tf.constant([1.])
b1 = tf.constant('hello,world')
c1 = np.arange(4)

# 是否是Tensor 的实例类型, isinstance 只能判断Tensor类型,不能判断派生类型
print(isinstance(a1, tf.Tensor))  # True
print(tf.is_tensor(a1))  # True
print(tf.is_tensor(c1))  # False

print(a1.dtype == tf.float32)  # True
print(b1.dtype == tf.string)  # True

# Convert
a2 = np.arange(5)
print(a2, a.dtype)  # [0 1 2 3 4] <dtype: 'int32'>
# Convert to tensor
# 将np 的列表转换为tensor对象
aa = tf.convert_to_tensor(a2)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
print(aa)
aa = tf.convert_to_tensor(a2, dtype=tf.int64)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
print(aa)

# Variable
a = tf.range(5)
b = tf.Variable(a)
print(b.dtype)
print(b.name)

b = tf.Variable(a, name='input_data')
print(b.name)
print(b.trainable)

# 无法判断Variable 为Tensor 对象
print(isinstance(b, tf.Tensor))  # False
print(isinstance(b, tf.Variable))  # True
print(tf.is_tensor(b))  # True
# 转换为numpy
print(b.numpy())  # [0 1 2 3 4]
