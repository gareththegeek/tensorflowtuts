import tensorflow as tf
import numpy as np
import math
import tflearn

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

input = tflearn.input_data(shape=[None], name="InputData")
linear = tflearn.layers.core.single_unit(
    input, activation="linear", name="Linear")

reg = tflearn.regression(
    linear,
    optimizer="sgd",
    loss="mean_square",
    metric="R2",
    learning_rate=0.01,
    name="Regression")

model = tflearn.DNN(reg)

model.fit(train_house_size_norm, train_price_norm, n_epoch=1000)

print("Training Complete")
print("Weights: W={0}, b={1}\n".format(
    model.get_weights(linear.W), model.get_weights(linear.b)))

print("Accuracy {0}".format(
    model.evaluate(test_house_size_norm, test_price_norm)))
