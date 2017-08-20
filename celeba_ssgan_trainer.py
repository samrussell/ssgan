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
from PIL import Image
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys, os

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

class CelebaSsganTrainer(base_trainer.BaseTrainer):
  img_rows = 64
  img_cols = 64
  img_channels = 3
  num_classes = 10

  def run(self):
    self.load_args()
    self.load_data()
    #shaped_values, shaped_labels = self.load_training_data()
    #testing_values, testing_labels = self.load_testing_data()
    #training_values, validation_values = self.split_data(shaped_values)
    #training_labels, validation_labels = self.split_data(shaped_labels)
    training_values = self.load_training_data()
    #training_values = training_values[:30000]
    #training_labels = training_labels[:30000]

    print('values shape:', training_values.shape)
    #print('values shape:', shaped_values.shape)
    #print(training_values.shape[0], 'training samples')
    #print(validation_values.shape[0], 'validation samples')

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

        # checkpoint data
        if self.commandline_args.save:
          self.encoder.save_weights("encoder.h5")
          self.decoder.save_weights("decoder.h5")
        self.save_results("test.png", training_values)
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
    #num_bits = 9
    total_num_bits = 99

    fixed_images = []
    for index in xrange(total_num_bits):
      vectors = []
      for vector in generated_vectors:
        vector_copy = np.copy(vector)
        vector_copy[index] = 1.0
        vectors.append(vector_copy)
      fixed_images.append(self.decoder.predict(np.array(vectors)))

    for num_bits in xrange(0, 99, 9):
      plt.figure(figsize=(10,10))

      for i in xrange(9):
        self.plot_image(input_images[i, :, :, :], i*10+1)
        for offset in xrange(9):
          self.plot_image(fixed_images[offset+num_bits][i, :, :, :], i*10+2+offset)
      plt.tight_layout()

      plt.savefig("test%d.png" % num_bits)
      plt.close('all')

  def build_models(self, input_shape):
    middle_neurons = 100

    self.encoder = Sequential()
    self.encoder.add(Conv2D(32, (3, 3), padding = 'same', input_shape=input_shape))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(32, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(MaxPooling2D(pool_size=(2, 2)))
    self.encoder.add(Conv2D(64, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(64, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(MaxPooling2D(pool_size=(2, 2)))
    self.encoder.add(Conv2D(128, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(128, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(MaxPooling2D(pool_size=(2, 2)))
    self.encoder.add(Conv2D(256, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(256, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(MaxPooling2D(pool_size=(2, 2)))
    self.encoder.add(Conv2D(512, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Conv2D(512, (3, 3), padding = 'same'))
    self.encoder.add(Activation(selu))
    self.encoder.add(Flatten())
    self.encoder.add(Dense(middle_neurons))
    self.encoder.add(Activation('sigmoid'))
    self.encoder.summary()

    self.decoder = Sequential()
    self.decoder.add(Dense(2*2*512, input_shape=(middle_neurons,)))
    self.decoder.add(Activation(selu))
    if keras.backend.image_data_format() == 'channels_first':
        self.decoder.add(Reshape([512, 2, 2]))
    else:    
        self.decoder.add(Reshape([2, 2, 512]))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(512, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(512, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(256, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(256, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(128, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(128, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(64, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(64, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2D(32, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(32, (3, 3), padding='same'))
    self.decoder.add(Activation(selu))
    self.decoder.add(Conv2D(3, (3, 3), padding='same'))
    self.decoder.add(Activation('sigmoid'))
    self.decoder.summary()

    self.autoencoder = Sequential()
    self.autoencoder.add(self.encoder)
    self.autoencoder.add(self.decoder)
    self.autoencoder.compile(loss='mean_squared_error',
                                  optimizer=Adam(lr=1e-4),
                                  metrics=['accuracy'])

  def load_data(self):
    images = []
    image_path = "celeba/img_align_celeba"

    filenames = os.listdir(image_path)
    filenames = np.random.choice(filenames, 10000, replace=False)
    for filename in filenames:
      if filename.endswith(".jpg"):
        image = Image.open("%s/%s" % (image_path, filename)).convert('RGB')
        image = image.crop((0, 20, 178, 198))
        image.thumbnail((64,64))
        image_data = np.asarray(image, dtype='float32')
        image_data /= 255.
        #test_image = image_data.transpose(2, 0, 1)
        #images.append(test_image)
        images.append(image_data)

    self.images = images

  def load_training_data(self):
    return np.array(self.images)

if __name__ == "__main__":
  CelebaSsganTrainer().run()
