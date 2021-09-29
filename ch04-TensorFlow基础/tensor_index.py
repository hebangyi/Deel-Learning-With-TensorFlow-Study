import tensorflow as tf

# tensor 切片,采集
a = tf.random.normal([4, 28, 28, 3])
print(a[1].shape)
print(a[1][2])
print(a[1, 2])
print(a[1][2][3])
print(a[1][2][3][2])

print(a[1][2][3:5])
print(a[1][2][3:-1])
print(a[1][2][0:-1])
print(a[:][:][0:1])  # 取出后面的特定维度

print(a[1][2][3:20:4])  # 其中4是step 步长
print(a[1][2][20:3:-1])  # 如果step是负数的话,则可以将采集的方式倒序

print(a[..., 0])  # 只显示最后一个维度
print(a[1, ..., 0])  # 只显示第一个和最后一个维度
print(a[1, 0, ..., 0])
print(a[:, tf.newaxis].shape)  # 增加一个维度
# 采样功能,在a的0维度上进行采样,索引是2,和4 总数是两个
print(tf.gather(a, axis=0, indices=[2, 4]).shape)
print(tf.gather(a, axis=1, indices=[2, 4]).shape)
# 采样功能,在a的0维度上进行采样,索引是2,1.3和0 总数是四个
print(tf.gather(a, axis=1, indices=[2, 1, 3, 0]).shape)

# 创建 4 35 8 的形状
b = tf.random.normal([4, 35, 8])
print(tf.gather_nd(b, [0]).shape)  # 采集b[0] 返回 (35, 8)
print(tf.gather_nd(b, [0, 1]).shape)  # 采集b[0][1] 返回 (8,)
print(tf.gather_nd(b, [0, 1, 2]))  # 采集b[0][1][2] 返回 数值
print(tf.gather_nd(b, [[0, 1, 2], [0, 1, 3], [1, 1, 1]]).shape)  # 数据采集后重新拼接组织,可以在外围添加一个维度 (3,)

print(tf.boolean_mask(b, mask=[True, True, False, False]).shape)  # 默认axis = 0 在第一个维度上采集两个数据
print(tf.boolean_mask(b, mask=[True, True, True, True, False, False, False, False],
                      axis=2).shape)  # 在第二个轴上采集4个数据
