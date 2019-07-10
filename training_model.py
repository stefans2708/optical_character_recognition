import keras
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
import constants

DATA_PATH = "/home/stefan/DataSets/chars_and_numbers_1/"

training_data = []

# Functions


# reading data from folders
# data is read as grayscale images
# data is normalized to get values between 0 and 1 by dividing with 255
# for each image in data array is placed category which is represented as number 1-a,27-A,and so on.
def load_data():
    for char_index in range(62):
        path = os.path.join(DATA_PATH, str(char_index + 1))

        for file in (os.listdir(path)):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img = img / 255
            training_data.append([img, char_index])


# data must be shuffled for better learning
def shuffle_data():
    random.shuffle(training_data)


# splits data into four arrays: x_train, x_test, y_train, y_test
# test data is 10% of whole data set
def split_data():
    x, y = [], []
    for features, labels in training_data:
        x.append(features)
        y.append(labels)
    return train_test_split(np.array(x), np.array(y), test_size=0.1, random_state=0)


# saves model architecture to json file and model weights to h5 file
def save_model_data():
    model_json = model.to_json()
    with open(constants.MODEL_ARCH_FILE_NAME, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(constants.MODEL_WEIGHTS_FILE_NAME)


# Main
load_data()
shuffle_data()
x_train, x_test, y_train, y_test = split_data()

# char_i = 10
# cv2.imshow(lower[y_test[char_i]], x_test[char_i])
# print(len(x_train), len(x_test), len(y_train), len(y_test))
# cv2.waitKey()
#
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(constants.IMG_SIZE, constants.IMG_SIZE)),
    keras.layers.Dense(4096, activation='tanh'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[keras.metrics.sparse_categorical_accuracy])
model.summary()

model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=2)

save_model_data()

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

y_out = model.predict(x_test, batch_size=32)
y_out = np.argmax(y_out, axis=1)
i = 0
plt.figure(figsize=(4, 4))
for img, out, exp in zip(x_test, y_out, y_test):
    if out != exp:
        plt.clf()
        plt.imshow(img)
        title = '{} misclassified as {}'.format(exp, out)
        plt.title(title)
        i += 1
        plt.savefig(os.path.join(os.getcwd(), 'prediction_errors', '{} ({}).png'.format(i, title)))
