# Implementation of Semi-Supervised Learning with Generative Adversarial Networks by Augustus Odena
# https://arxiv.org/pdf/1606.01583.pdf
# Also draws on UNSUPERVISED AND SEMI-SUPERVISED LEARNING WITH CATEGORICAL GENERATIVE ADVERSARIAL NETWORKS
# by Jost Tobias Springenberg
# https://arxiv.org/pdf/1511.06390.pdf
# Then based on STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
# https://arxiv.org/pdf/1412.6806.pdf
# Builds on [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
# block building code from https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# Code (c) Sam Russell 2017
# this reaches 81% val accuracy and 99.9% training accuracy (overfitting quite hard)
# it produces some passable images in the process though (not super high quality)

import base_trainer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input
from keras import layers
from keras.layers.core import Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.activations import *
from keras.utils import to_categorical
from keras.datasets import cifar10
import keras

def selu(x):
  """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
  # Arguments
      x: A tensor or variable to compute the activation function for.
  # References
      - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)

class CifarSsganTrainer(base_trainer.BaseTrainer):
  img_rows = 32
  img_cols = 32
  img_channels = 3
  num_classes = 10

  def block(self, num_filters, num_layers, kernel_size, strides, input_tensor):
    x = Conv2D(num_layers, (1, 1), strides=strides)(input_tensor)
    x = Activation(selu)(x)
    
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Activation(selu)(x)

    x = Conv2D(num_filters*4, (1, 1))(x)

    shortcut = Conv2D(num_filters*4, (1, 1), strides=strides,
                      )(input_tensor)

    x = layers.add([x, shortcut])
    x = Activation(selu)(x)

    return x

  def build_models(self, input_shape):
    image_input = Input(shape=input_shape)
    input_tensor = image_input

    # first layer
    x = Conv2D(64, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(256, (1, 1), padding="same")(x)
    shortcut = Conv2D(256, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(64, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    shortcut = Conv2D(256, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)

    # second layer
    x = Conv2D(128, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    shortcut = Conv2D(512, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(128, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    shortcut = Conv2D(512, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(128, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    shortcut = Conv2D(512, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)

    # third layer
    x = Conv2D(256, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(1024, (1, 1), padding="same")(x)
    shortcut = Conv2D(1024, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(256, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(1024, (1, 1), padding="same")(x)
    shortcut = Conv2D(1024, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(256, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(1024, (1, 1), padding="same")(x)
    shortcut = Conv2D(1024, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)

    # fourth layer
    x = Conv2D(512, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(2048, (1, 1), padding="same")(x)
    shortcut = Conv2D(2048, (1, 1), padding="same", strides=(2, 2))(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(512, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(2048, (1, 1), padding="same")(x)
    shortcut = Conv2D(2048, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)
    x = Conv2D(512, (1, 1), padding="same")(input_tensor)
    x = Activation(selu)(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = Activation(selu)(x)
    x = Conv2D(2048, (1, 1), padding="same")(x)
    shortcut = Conv2D(2048, (1, 1), padding="same")(input_tensor)
    x = layers.add([x, shortcut])
    input_tensor = Activation(selu)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation(selu)(x)
    x = Dense(1+self.num_classes, activation='linear')(x)
    self.discriminator = Model(image_input, x)
    self.discriminator.summary()

    self.generator = Sequential()
    self.generator.add(Dense(1000, input_shape=(100,)))
    self.generator.add(Activation(selu))
    self.generator.add(Dropout(0.5))
    self.generator.add(Dense(1000))
    self.generator.add(Activation(selu))
    self.generator.add(Dropout(0.5))
    self.generator.add(Dense(8*8*192))
    self.generator.add(Activation(selu))
    self.generator.add(Dropout(0.5))
    if keras.backend.image_data_format() == 'channels_first':
        self.generator.add(Reshape([192, 8, 8]))
    else:    
        self.generator.add(Reshape([8, 8, 192]))
    self.generator.add(Conv2D(192, (1, 1), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(192, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(0.2))
    self.generator.add(Conv2D(192, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(192, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(192, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(0.2))
    self.generator.add(Conv2D(96, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(96, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(96, (3, 3), padding='same'))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation(selu))
    self.generator.add(Dropout(0.2))
    self.generator.add(Conv2D(3, (3, 3), padding='same'))
    self.generator.add(Activation('sigmoid'))
    self.generator.summary()

    self.generator.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=1e-5, beta_1=0.5),
                           metrics=['accuracy'])

    self.discriminator.compile(loss='mean_squared_error',
                               optimizer=Adam(lr=1e-5, beta_1=0.5),
                               metrics=['accuracy'])

    self.real_image_model = self.discriminator

    self.fake_image_model = Sequential()
    self.fake_image_model.add(self.generator)
    self.discriminator.trainable = False
    self.fake_image_model.add(self.discriminator)
    self.fake_image_model.compile(loss='mean_squared_error',
                                  optimizer=Adam(lr=1e-5, beta_1=0.5),
                                  metrics=['accuracy'])

  def load_data(self):
    self.cifar_data = cifar10.load_data()

  def load_training_data(self):
    #training_dataframe = pandas.read_csv(self.commandline_args.train)
    #values = training_dataframe.values[:,1:]
    #labels = training_dataframe.values[:,0]
    (X_train, y_train), (X_test, y_test) = self.cifar_data
    
    shaped_labels = to_categorical(y_train, self.num_classes+1)
    scaled_values = self.scale_values(X_train)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

  def load_testing_data(self):
    #testing_dataframe = pandas.read_csv(self.commandline_args.test)
    #values = testing_dataframe.values
    
    (X_train, y_train), (X_test, y_test) = self.cifar_data
    shaped_labels = to_categorical(y_test, self.num_classes+1)
    scaled_values = self.scale_values(X_test)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

if __name__ == "__main__":
  CifarSsganTrainer().run()
