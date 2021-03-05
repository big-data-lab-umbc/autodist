import tensorflow as tf
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications import InceptionV3


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class ModelManager:

    def __init__(self, model, input_shape, classes):
        self.input_shape = input_shape#(input_shape[0], input_shape[1], 1)

        possible_model = {
            "Small": self.getSmallModel(),
            "MobileNetV2": MobileNetV2(),
            "Resnet50":  ResNet50(),
            "VGG": self.VGG(),
            "DenseNet": DenseNet121(),
            "Cifarnet": self.cifarnet(),
            "Inception": InceptionV3()
        }

        assert model in possible_model.keys(), "Selected model not available"
        self.model = possible_model[model]

    def getSmallModel(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        return model

    def VGG(self):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        return model

    def cifarnet(self):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((3, 3)))

        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((3, 3)))
        
        model.add(Flatten())
        model.add(Dense(384, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(192, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        return model