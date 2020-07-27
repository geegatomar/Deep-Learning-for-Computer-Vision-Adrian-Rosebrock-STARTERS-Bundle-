from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras import backend as K

class LeNet:
    @staticmethod
    def build(height, width, depth, classes):
        inputShape = (height, width, depth)
        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)
        model = Sequential()

        model.add(Conv2D(20, (5, 5), input_shape=inputShape, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(classes, activation="softmax"))

        return model

