import tensorflow as tf

a = tf.fill([2, 2], 3.)
print(tf.norm(a))  # a 的二范数

b = tf.random.normal([4, 5, 2])
print(tf.reduce_min(b))
print(tf.reduce_max(b))
print(tf.reduce_mean(b))

# axis 在一维上的最大最小值
print(tf.reduce_min(b, axis=1))
print(tf.reduce_max(b, axis=1))
print(tf.reduce_mean(b, axis=1))

print(tf.argmax(b))  # 求最大值所在的位置索引,默认 axis = 0
print(tf.argmax(b, axis=2))

c = tf.constant([1, 2, 3, 3, 5])
d = tf.range(5)
e = tf.equal(c, d)
print(e)  # 只能相同维度的元素做比较 tf.Tensor([False False False False False], shape=(5,), dtype=bool)
print(tf.cast(e, dtype=tf.int32))  # 转换数值

print(tf.unique(c))  # 去除重复元素

# 张量限幅
f = tf.range(9)
print(tf.maximum(f, 5))  # 最小值的不能低于5
print(tf.minimum(f, 5))  # 最大值的不能低于5
print(tf.clip_by_value(f, 2, 8))  # 最大值和最小值在2到8之间

g = tf.random.normal([2, 2], mean=10)
print(g)
tf.norm(g)
aa = tf.clip_by_norm(g, 15)  # 将矩阵的值进行缩放,将二范数的值约束为15
print(aa)
print(tf.norm(aa))

# h = tf.range(9)
# bb, _ = tf.clip_by_global_norm([1,2,3,4], 15)
# print(bb)
