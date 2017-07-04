# Code (c) Sam Russell 2017
import pandas
import argparse
import numpy as np
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

class BaseTrainer:
  batch_size = 128
  num_classes = 10
  epochs = 12
  validation_percentage = 0.99
  img_rows = 28
  img_cols = 28
  img_channels = 1

  def __init__(self):
    # set up commandline arguments
    #self.parser = argparse.ArgumentParser()
    #self.parser.add_argument('--train', help='training data CSV', required=True)
    #self.parser.add_argument('--test', help='test data CSV', required=True)
    #self.parser.add_argument('--output', help='output data CSV', required=True)
    pass

  def run(self):
    #self.load_args()
    self.load_mnist_data()
    shaped_values, shaped_labels = self.load_training_data()
    testing_values, testing_labels = self.load_testing_data()
    training_values, validation_values = self.split_data(shaped_values)
    training_labels, validation_labels = self.split_data(shaped_labels)

    print('values shape:', shaped_values.shape)
    print(training_values.shape[0], 'training samples')
    print(validation_values.shape[0], 'validation samples')

    self.build_models(input_shape=training_values.shape[1:])

    # training
    # do this in a loop
    num_samples = training_values.shape[0]
    num_fakes = int(num_samples / self.num_classes)

    for i in xrange(self.epochs):
      fake_values = np.random.uniform(0,1,size=[num_fakes,100])
      fake_labels = to_categorical(np.full((num_fakes, 1), self.num_classes), self.num_classes+1)
      self.real_image_model.fit(training_values, training_labels,
                batch_size=self.batch_size,
                epochs=1,
                verbose=1,
                validation_data=(validation_values, validation_labels))
      self.fake_image_model.fit(fake_values, fake_labels,
                batch_size=self.batch_size,
                epochs=1,
                verbose=1)
      self.save_results("test_%d.png" % i)

    #self.test_results(testing_values, testing_labels)

  def save_results(self, filename):
    # save some samples
    fake_values = np.random.uniform(0,1,size=[16,100])
    images = self.generator.predict(fake_values)
    plt.figure(figsize=(10,10))

    for i in range(images.shape[0]):
      plt.subplot(4, 4, i+1)
      image = images[i, :, :, :]
      image = np.reshape(image, [self.img_rows, self.img_cols])
      plt.imshow(image, cmap='gray')
      plt.axis('off')
    plt.tight_layout()

    plt.savefig(filename)
    plt.close('all')

  #def test_results(self, testing_values, testing_labels):
    #predictions = self.model.predict(testing_values)
    #df = pandas.DataFrame(data=np.argmax(predictions, axis=1), columns=['Label'])
    #df.insert(0, 'ImageId', range(1, 1 + len(df)))

    # save results
    #df.to_csv(self.commandline_args.output, index=False)

  #def load_args(self):
  #  self.commandline_args = self.parser.parse_args()

  def load_mnist_data(self):
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

  def scale_values(self, values):
    return values.astype('float32') / 255

  def reshape_values(self, values):
    # TODO make it work when data comes pre-shaped
    if K.image_data_format() == 'channels_first':
        reshaped_values = values.reshape(values.shape[0], self.img_channels, self.img_rows, self.img_cols)
    else:
        reshaped_values = values.reshape(values.shape[0], self.img_rows, self.img_cols, self.img_channels)

    return reshaped_values

  def split_data(self, data):
    landmark = int(data.shape[0] * self.validation_percentage)
    return data[:landmark], data[landmark:]

  def build_models(self, input_shape):
    raise NotImplementedError("Must be implemented by subclass")
