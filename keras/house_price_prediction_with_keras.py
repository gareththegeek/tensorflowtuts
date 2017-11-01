import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import Sequential
from keras.layers.core import Dense, Activation

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

model = Sequential()
model.add(Dense(1, input_shape=(1,), init="uniform", activation="linear"))
model.compile(loss="mean_squared_error", optimizer="sgd")

model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)

score = model.evaluate(test_house_size_norm, test_price_norm)
print("\nloss on test : {0}".format(score))
