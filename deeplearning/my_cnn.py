import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# 读入MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# compute the accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs : v_xs, keep_prob : 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs : v_xs, ys : v_ys, keep_prob : 1})
    return result

# define weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# define biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# cov2d 步长 [1, 1]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# max_pool
def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# define placeholder for dropout
keep_prob = tf.placeholder(tf.float32)

# 把数据转换成 适合 nn 输入的数据格式 ，-1代表先不考虑输入的图片例子多少个维度
x_image = tf.reshape(xs, [-1, 28, 28, 1])# 转换为图像的形式

# conv1 layer
# patch 5x5 ,channel is 1 ， output 32 featuremap
with tf.name_scope('conv_relu_1') as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])#patch 5*5,in_size 1,out size32
    b_conv1 = bias_variable([32])

# structure is 28x28x32
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#output size 28*28*32
with tf.name_scope('pool_1') as scope:
    h_pool1 = max_poo_2x2(h_conv1) #output size 14*14*32

# conv2 layer
with tf.name_scope('conv_relu_2') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])#patch 5*5,in_size 32,out size64
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#output size 14*14*64
with tf.name_scope('pool_2') as scope:
    h_pool2 = max_poo_2x2(h_conv2) #output size 7*7*64

## func1 layer
with tf.name_scope('fc_rule_1') as scope:
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
## [n_samples,7,7,64]->>[n_samples,7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
with tf.name_scope('dropout_1') as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer
with tf.name_scope('fc_softmax_2') as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('loss') as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
with tf.name_scope('AdamOptimizer') as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

# important step of initial
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # 获取小批量数据，100个一次训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))