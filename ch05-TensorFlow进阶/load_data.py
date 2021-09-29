import tensorflow as tf
import tensorflow.keras.datasets as datasets


def prepare_mnist_features_and_labels(x, y):
    """
    数据处理函数
    :param x:
    :param y:
    :return:
    """
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def mnist_dataset():
    (x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)  # 根据值找到索引的位置,在相应的位置上赋值1
    y_val = tf.one_hot(y_val, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y))  # 加载数据x y
    ds = ds.map(prepare_mnist_features_and_labels)  # 设置数据预处理函数
    ds = ds.shuffle(60000).batch(100)  # 打乱数据集,并设置数据集批量

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_mnist_features_and_labels)
    ds_val = ds_val.shuffle(10000).batch(100)
    return ds, ds_val
