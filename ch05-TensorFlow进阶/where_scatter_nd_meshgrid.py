import tensorflow as tf

# where
a = tf.random.normal([3, 3])
print(a)

mask = a > 0
print(mask)
mask_val = tf.boolean_mask(a, mask)
print(mask_val)
indices = tf.where(mask)  # where等于一个参数的时候,返回等于true的索引值
print(indices)
print(tf.gather_nd(a, indices))  # 根据索引值,去除数据,并将数据拼接起来

A = tf.ones([3, 3])
B = tf.zeros([3, 3])
C = tf.where(mask, A, B)  # 三个参数的where,如果值是True,则选择左边A中对应矩阵列的值,如果值是False,则选择矩阵右边B中对应矩阵列的值
print(C)

# scatter_nd
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])  # 值分别对应index=4, index=3, index=1, index=7
shape = tf.constant([8])
print(shape)
aa = tf.scatter_nd(indices, updates, shape)  # 在shape上填充数值
print(aa)

indices_1 = tf.constant([[0], [2]])
updates_1 = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                         [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]])
shape_1 = tf.constant([4, 4, 4])
# 更新的数据
scatter_shape = tf.scatter_nd(indices_1, updates_1, shape_1)
print(scatter_shape)
