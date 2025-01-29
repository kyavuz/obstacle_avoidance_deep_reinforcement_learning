# models.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class DuelingDQN(Model):
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        self.conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")
        self.conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")
        self.conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.value = layers.Dense(1, activation=None)
        self.advantage = layers.Dense(action_size, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
