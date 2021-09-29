import numpy as np
import tensorflow as tf

# 根据数据进行初始化
print(tf.convert_to_tensor(np.ones([2, 3])))  # tf.Tensor([[1. 1. 1.] [1. 1. 1.]], shape=(2, 3), dtype=float64)
print(tf.convert_to_tensor(np.zeros([2, 3])))  # tf.Tensor( [[0. 0. 0.] [0. 0. 0.]], shape=(2, 3), dtype=float64)
print(tf.convert_to_tensor([1, 2]))  # tf.Tensor([1 2], shape=(2,), dtype=int32)
print(tf.convert_to_tensor([1, 2.]))  # tf.Tensor([1. 2.], shape=(2,), dtype=float32)
print(tf.convert_to_tensor([[1], [2.]]))  # tf.Tensor([[1.] [2.]], shape=(2, 1), dtype=float32)

# 初始化为0,传入的是shape 维度
print(tf.zeros([]))  # tf.Tensor(0.0, shape=(), dtype=float32)
print(tf.zeros([1]))  # tf.Tensor([0.], shape=(1,), dtype=float32)
print(tf.zeros([2, 2]))  # tf.Tensor([[0. 0.] [0. 0.]], shape=(2, 2), dtype=float32)
print(tf.zeros([2, 3,
                3]))  # tf.Tensor([[[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]] [[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]]], shape=(2, 3, 3), dtype=float32)
a = tf.zeros([2, 3, 3])
print(tf.zeros_like([2, 3, 3]))  # tf.Tensor([[[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]]
print(tf.zeros(a.shape))  # tf.Tensor([[[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]]

print(tf.ones(1))
print(tf.ones([]))
print(tf.ones([2]))
print(tf.ones([2, 3]))
print(tf.ones_like(a))

# 使用fill函数确定shape和填充值
print(tf.fill([2, 2], 0))
print(tf.fill([2, 2], 0))
print(tf.fill([2, 2], 1))
print(tf.fill([2, 2], 9))

# 随机,正态分布,均值是1 方差是1
print(tf.random.normal([2, 2], mean=1, stddev=1))
# 随机,截断的方式,正态分布,均值是1 方差是1
print(tf.random.truncated_normal([2, 2], mean=1, stddev=1))  # 截断函数有助于数据无限趋于0 梯度消失

# 均匀分布, 从0 到 1中间进行采样
print(tf.random.uniform([2, 2], minval=0, maxval=1))
print(tf.random.uniform([3, 4], minval=0, maxval=100))

idx = tf.range(10)  # 生成一个一维的数据
idx = tf.random.shuffle(idx)  # 一维数据进行随机
print(idx)  # [4 7 9 8 5 1 0 3 6 2]


