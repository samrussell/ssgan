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
import keras

class SsGanTrainer(base_trainer.BaseTrainer):
  def build_models(self, input_shape):
    self.discriminator = Sequential()
    self.discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding = 'same', activation='relu', input_shape=input_shape))
    self.discriminator.add(LeakyReLU(0.2))
    self.discriminator.add(Dropout(0.5))
    self.discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding = 'same', activation='relu'))
    self.discriminator.add(LeakyReLU(0.2))
    self.discriminator.add(Dropout(0.5))
    self.discriminator.add(Conv2D(256, (5, 5), strides=(2, 2), padding = 'same', activation='relu'))
    self.discriminator.add(LeakyReLU(0.2))
    self.discriminator.add(Dropout(0.5))
    # 7x7 for MNIST
    #H = Conv2D(512, (5, 5), strides=(2, 2), padding = 'same', activation='relu')(H)
    #H = LeakyReLU(0.2)(H)
    #H = Dropout(0.5)(H)
    self.discriminator.add(Flatten())
    self.discriminator.add(Dense(1+self.num_classes,activation='softmax'))
    self.discriminator.summary()

    self.generator = Sequential()
    self.generator.add(Dense(7*7*256, input_shape=(100,)))
    #self.generator.add(BatchNormalization())
    self.generator.add(Activation('relu'))
    if keras.backend.image_data_format() == 'channels_first':
        self.generator.add(Reshape([256, 7, 7]))
    else:    
        self.generator.add(Reshape([7, 7, 256]))
    self.generator.add(Dropout(0.5))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Conv2D(128, (5, 5), padding='same'))
    self.generator.add(BatchNormalization())
    self.generator.add(Activation('relu'))
    self.generator.add(Dropout(0.5))
    self.generator.add(UpSampling2D(size=(2, 2)))
    self.generator.add(Conv2D(256, (5, 5), padding='same'))
    self.generator.add(BatchNormalization())
    self.generator.add(Activation('relu'))
    # we're ignoring input shape - just assuming it's 7,7,1
    self.generator.add(Conv2D(1, (5, 5), padding='same'))
    self.generator.add(Activation('sigmoid'))
    self.generator.summary()

    self.real_image_model = Sequential()
    self.real_image_model.add(self.discriminator)
    self.real_image_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))

    self.fake_image_model = Sequential()
    self.fake_image_model.add(self.generator)
    self.discriminator.trainable = False
    self.fake_image_model.add(self.discriminator)
    self.fake_image_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))

if __name__ == "__main__":
  SsGanTrainer().run()