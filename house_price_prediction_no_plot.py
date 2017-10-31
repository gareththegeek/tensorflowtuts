import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
house_price = house_size * 100.0 + \
    np.random.randint(low=20000, high=70000, size=num_house)

def normalize(array):
    return (array - array.mean()) / array.std()


num_train_samples = math.floor(num_house * 0.7)

# Training Data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# Test Data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

print(tf_house_size)
print(tf_price)

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

print(tf_size_factor)
print(tf_price_offset)

tf_price_pred = tf.add(tf.multiply(
    tf_size_factor, tf_house_size), tf_price_offset)

tf_cost = tf.reduce_sum(
    tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):
        for(x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={
                             tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(
        tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()
