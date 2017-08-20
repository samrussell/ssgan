# Implementation of Semi-Supervised Learning with Generative Adversarial Networks by Augustus Odena
# https://arxiv.org/pdf/1606.01583.pdf
# Also draws on UNSUPERVISED AND SEMI-SUPERVISED LEARNING WITH CATEGORICAL GENERATIVE ADVERSARIAL NETWORKS
# by Jost Tobias Springenberg
# https://arxiv.org/pdf/1511.06390.pdf
# Code (c) Sam Russell 2017

import base_trainer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.activations import *
from keras.utils import to_categorical
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt

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

class Mnist2SsganTrainer(base_trainer.BaseTrainer):
  img_rows = 28
  img_cols = 28
  img_channels = 1
  num_classes = 10

  def run(self):
    self.load_args()
    self.load_data()
    shaped_values, shaped_labels = self.load_training_data()
    testing_values, testing_labels = self.load_testing_data()
    training_values, validation_values = self.split_data(shaped_values)
    training_labels, validation_labels = self.split_data(shaped_labels)
    #training_values = training_values[:30000]
    #training_labels = training_labels[:30000]

    print('values shape:', shaped_values.shape)
    print(training_values.shape[0], 'training samples')
    print(validation_values.shape[0], 'validation samples')

    self.build_models(input_shape=training_values.shape[1:])

    if self.commandline_args.load:
      self.encoder.load_weights("encoder.h5")
      self.decoder.load_weights("decoder.h5")

    if self.commandline_args.train:
      while True:
        self.autoencoder.fit(training_values, training_values,
              batch_size=self.batch_size,
              epochs=1,
              verbose=1)

        self.save_results("test.png", training_values)
        # checkpoint data
        if self.commandline_args.save:
          self.encoder.save_weights("encoder.h5")
          self.decoder.save_weights("decoder.h5")
    else:
      self.save_results("test.png", training_values)

  def plot_image(self, image, index):
    if self.img_channels == 1:
      image = np.reshape(image, [self.img_rows, self.img_cols])
    elif K.image_data_format() == 'channels_first':
      image = image.transpose(1,2,0)
    # implicit no need to transpose if channels are last
    plt.subplot(10, 10, index)
    plt.imshow(image, cmap='gray')
    plt.axis('off')


  def save_results(self, filename, training_values):
    # save some samples
    input_images = training_values[np.random.choice(training_values.shape[0], 10, replace=False)]
    generated_vectors = self.encoder.predict(input_images)
    generated_images = self.decoder.predict(generated_vectors)
    num_bits = 9

    fixed_images = []
    for index in xrange(num_bits):
      vectors = []
      for vector in generated_vectors:
        vector_copy = np.copy(vector)
        vector_copy[index] = 1.0
        vectors.append(vector_copy)
      fixed_images.append(self.decoder.predict(np.array(vectors)))

    plt.figure(figsize=(10,10))

    for i in range(input_images.shape[0]):
      self.plot_image(input_images[i, :, :, :], i*(num_bits+1)+1)
      for offset in xrange(num_bits):
        self.plot_image(fixed_images[offset][i, :, :, :], i*(num_bits+1)+2+offset)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close('all')

  def build_models(self, input_shape):
    middle_neurons = 10

    self.encoder = Sequential()
    self.encoder.add(Conv2D(64, (5, 5), strides=(2, 2), padding = 'same', input_shape=input_shape))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(128, (5, 5), strides=(2, 2), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Flatten())
    self.encoder.add(Dense(middle_neurons))
    self.encoder.add(Activation('sigmoid'))
    self.encoder.summary()

    self.decoder = Sequential()
    self.decoder.add(Dense(7*7*128, input_shape=(middle_neurons,)))
    self.decoder.add(Activation(selu))
    if keras.backend.image_data_format() == 'channels_first':
        self.decoder.add(Reshape([128, 7, 7]))
    else:    
        self.decoder.add(Reshape([7, 7, 128]))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(64, (5, 5), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(1, (5, 5), padding='same'))
    self.decoder.add(Activation('sigmoid'))
    self.decoder.summary()

    self.autoencoder = Sequential()
    self.autoencoder.add(self.encoder)
    self.autoencoder.add(self.decoder)
    self.autoencoder.compile(loss='mean_squared_error',
                                  optimizer=Adam(lr=1e-4),
                                  metrics=['accuracy'])

  def load_data(self):
    self.mnist_data = mnist.load_data()

  def load_training_data(self):
    #training_dataframe = pandas.read_csv(self.commandline_args.train)
    #values = training_dataframe.values[:,1:]
    #labels = training_dataframe.values[:,0]
    (X_train, y_train), (X_test, y_test) = self.mnist_data
    
    shaped_labels = to_categorical(y_train, self.num_classes+1)
    scaled_values = self.scale_values(X_train)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

  def load_testing_data(self):
    #testing_dataframe = pandas.read_csv(self.commandline_args.test)
    #values = testing_dataframe.values
    
    (X_train, y_train), (X_test, y_test) = self.mnist_data
    shaped_labels = to_categorical(y_test, self.num_classes+1)
    scaled_values = self.scale_values(X_test)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

if __name__ == "__main__":
  Mnist2SsganTrainer().run()
