import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logPath = "./tb_logs/"


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")
    tf.summary.image("input_img", x_image, 5)


def weight_variable(shape):
    with tf.name_scope("weights"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        variable_summaries(initial)
        return tf.Variable(initial, name="weight")


def bias_variable(shape):
    with tf.name_scope("biases"):
        initial = tf.constant(0.1, shape=shape)
        variable_summaries(initial)
        return tf.Variable(initial, name="bias")


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="pool")


def relu(input, weights, bias):
    conv_wx_b = conv2d(input, weights) + bias
    tf.summary.histogram("conv_wx_b", conv_wx_b)
    result = tf.nn.relu(conv_wx_b, name="relu")
    tf.summary.histogram("conv", result)
    return result


with tf.name_scope("Conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = relu(x_image, W_conv1, b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Conv2"):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = relu(h_pool1, W_conv2, b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("FC"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Readout"):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

summarise_all = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

tbWriter = tf.summary.FileWriter(logPath, sess.graph)

import time
num_steps = 2000
display_every = 100

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    _, summary = sess.run(
        [train_step, summarise_all],
        feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5
        })

    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0
        })
        end_time = time.time()
        print(
            "step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".
            format(i, end_time - start_time, train_accuracy * 100.0))
        tbWriter.add_summary(summary, i)

end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(
    i + 1, end_time - start_time))

# This doesn't work on GPU as runs out of memory
# print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={ \
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0 \
# }) * 100.0))

accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for i in range(10):
    testSet = mnist.test.next_batch(50)
    good += accuracy_sum.eval(feed_dict={
        x: testSet[0],
        y_: testSet[1],
        keep_prob: 1.0
    })
    total += testSet[0].shape[0]
print("test accuracy {0:.3f}%".format(good / total * 100.0))

sess.close()
