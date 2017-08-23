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
import keras.backend as K
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
      self.discriminator.load_weights("discriminator.h5")
      self.generator.load_weights("generator.h5")

    num_samples = 1000
    zero_vector = np.repeat([[0, 1]], num_samples, axis=0)
    one_vector = np.repeat([[1, 0]], num_samples, axis=0)
    labels_for_discriminator = np.concatenate((zero_vector, one_vector), axis=0)
    labels_for_generator = np.concatenate((one_vector, one_vector), axis=0)

    if self.commandline_args.train:
      while True:
        real_sample = training_values[np.random.choice(training_values.shape[0], num_samples, replace=False)]
        vectors = np.random.rand(num_samples*2, 100)
        print("Generating fake images")
        fake_sample = self.generator.predict(vectors[:1000], verbose=1)

        print("Training discriminator")
        # labels are 00000...111111
        # values are fakefakefakefake...realrealrealreal
        samples = np.concatenate((fake_sample, real_sample), axis=0)
        self.discriminator.fit(samples, labels_for_discriminator,
              batch_size=self.batch_size,
              epochs=1,
              verbose=1)
        print("Training generator")
        # labels are 111111
        # values are fakefakefakefake
        self.generator_trainer.fit(vectors, labels_for_generator,
              batch_size=self.batch_size,
              epochs=1,
              verbose=1)

        # checkpoint data
        if self.commandline_args.save:
          self.discriminator.save_weights("discriminator.h5")
          self.generator.save_weights("generator.h5")
        if self.commandline_args.demo:
          print("Saving demo")
          self.save_results("test.png", fake_sample)
    elif self.commandline_args.demo:
      print("Saving demo")
      self.save_results("test.png", fake_sample)

  def plot_image(self, image, index):
    if self.img_channels == 1:
      image = np.reshape(image, [self.img_rows, self.img_cols])
    elif K.image_data_format() == 'channels_first':
      image = image.transpose(1,2,0)
    # implicit no need to transpose if channels are last
    plt.subplot(10, 10, index)
    plt.imshow(image, cmap='gray')
    plt.axis('off')


  def save_results(self, filename, input_images):
    # save some samples
    plt.figure(figsize=(10,10))

    for i in xrange(100):
      self.plot_image(input_images[i, :, :, :], i+1)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close('all')

  def build_models(self, input_shape):
    middle_neurons = 100
    dropout_rate = 0.2

    self.discriminator = Sequential()
    self.discriminator.add(Conv2D(32, (3, 3), padding = 'same', input_shape=input_shape))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Conv2D(32, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    self.discriminator.add(Dropout(dropout_rate))
    self.discriminator.add(Conv2D(64, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Conv2D(64, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    self.discriminator.add(Dropout(dropout_rate))
    self.discriminator.add(Conv2D(128, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Conv2D(128, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    self.discriminator.add(Dropout(dropout_rate))
    self.discriminator.add(Conv2D(256, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Conv2D(256, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    self.discriminator.add(Dropout(dropout_rate))
    self.discriminator.add(Conv2D(512, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Conv2D(512, (3, 3), padding = 'same'))
    self.discriminator.add(Activation(selu))
    self.discriminator.add(Dropout(dropout_rate))
    self.discriminator.add(Flatten())
    self.discriminator.add(Dense(1000))
    self.discriminator.add(Activation('sigmoid'))
    self.discriminator.add(Dense(2))
    self.discriminator.add(Activation('softmax'))
    self.discriminator.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=1e-5))
    self.discriminator.summary()

    self.generator = Sequential()
    self.generator.add(Dense(2*2*512, input_shape=(middle_neurons,)))
    self.generator.add(Activation(selu))
    if keras.backend.image_data_format() == 'channels_first':
        self.generator.add(Reshape([512, 2, 2]))
    else:    
        self.generator.add(Reshape([2, 2, 512]))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Conv2D(512, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(512, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(dropout_rate))
    self.generator.add(Conv2D(256, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(256, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(dropout_rate))
    self.generator.add(Conv2D(128, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(128, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(dropout_rate))
    self.generator.add(Conv2D(64, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(64, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Dropout(dropout_rate))
    self.generator.add(Conv2D(32, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Conv2D(32, (3, 3), padding='same'))
    self.generator.add(Activation(selu))
    self.generator.add(Dropout(dropout_rate))
    self.generator.add(Conv2D(3, (3, 3), padding='same'))
    self.generator.add(Activation('sigmoid'))
    self.generator.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=1e-5))
    self.generator.summary()

    self.discriminator.trainable = False
    gan_input = Input(shape=(middle_neurons,))
    x = self.generator(gan_input)
    gan_output = self.discriminator(x)
    self.generator_trainer = Model(gan_input, gan_output)
    self.generator_trainer.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=1e-5))
    self.generator_trainer.summary()

  def load_data(self):
    images = []
    image_path = "celeba/img_align_celeba"

    filenames = os.listdir(image_path)
    if self.commandline_args.train:
      filenames = np.random.choice(filenames, 10000, replace=False)
    else:
      filenames = np.random.choice(filenames, 100, replace=False)
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
