# Code (c) Sam Russell 2017
import pandas
import argparse
import numpy as np
import keras
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class BaseTrainer:
  batch_size = 128
  epochs = 1200
  validation_percentage = 0.99

  def __init__(self):
    # set up commandline arguments
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--save', help='save to file (default false)', default=False, action='store_true')
    self.parser.add_argument('--load', help='load from file (default false)', default=False, action='store_true')
    #self.parser.add_argument('--train', help='training data CSV', required=True)
    #self.parser.add_argument('--test', help='test data CSV', required=True)
    #self.parser.add_argument('--output', help='output data CSV', required=True)
    pass

  def run(self):
    self.load_args()
    self.load_data()
    shaped_values, shaped_labels = self.load_training_data()
    testing_values, testing_labels = self.load_testing_data()
    training_values, validation_values = self.split_data(shaped_values)
    training_labels, validation_labels = self.split_data(shaped_labels)
    #training_values = training_values[:1000]
    #training_labels = training_labels[:1000]

    print('values shape:', shaped_values.shape)
    print(training_values.shape[0], 'training samples')
    print(validation_values.shape[0], 'validation samples')

    self.build_models(input_shape=training_values.shape[1:])

    if self.commandline_args.load:
      self.generator.load_weights("generator.h5")
      self.discriminator.load_weights("discriminator.h5")

    # training
    # do this in a loop
    num_samples = training_values.shape[0]
    #num_fakes = int(num_samples / self.num_classes)
    #num_fakes = num_samples
    num_to_train = 24000
    num_fakes_for_discriminator = int(num_to_train / 4)
    num_fakes_for_generator = num_fakes_for_discriminator + num_to_train
    for i in xrange(self.epochs):
      for offset in range(0, num_samples, num_to_train)[:-1]:
        # we want the discriminator to guess the fakes
        print("generating images")
        training_value_batch = training_values[offset:offset+num_to_train]
        training_label_batch = training_labels[offset:offset+num_to_train]
        fake_categories = np.random.choice(self.num_classes,num_fakes_for_generator)
        fake_vectors = to_categorical(fake_categories, self.num_classes+1)
        random_value_part = np.random.uniform(0,1,size=[num_fakes_for_generator,100-(self.num_classes+1)])
        fake_values = np.concatenate((fake_vectors, random_value_part), axis=1)
        fake_labels = to_categorical(np.full((num_fakes_for_generator, 1), self.num_classes), self.num_classes+1)
        fake_images = self.generator.predict(fake_values[:num_fakes_for_discriminator], verbose=0)

        print("training discriminator")
        self.discriminator.trainable = True
        self.real_image_model.fit(np.concatenate((training_value_batch, fake_images)),
                  np.concatenate((training_label_batch, fake_labels[:num_fakes_for_discriminator])),
                  batch_size=self.batch_size,
                  epochs=1,
                  verbose=1,
                  validation_data=(validation_values, validation_labels))

        # we want the discriminator to guess the category we injected
        print("training generator")
        self.discriminator.trainable = False
        self.fake_image_model.fit(fake_values, fake_vectors,
                  batch_size=self.batch_size,
                  epochs=1,
                  verbose=1)
        self.save_results("test_%d_%d.png" % (i, offset))
      # checkpoint data
      if self.commandline_args.save:
        self.generator.save_weights("generator.h5")
        self.discriminator.save_weights("discriminator.h5")

    #self.test_results(testing_values, testing_labels)

  def save_results(self, filename):
    # save some samples
    fake_categories = np.random.choice(self.num_classes,16)
    fake_vectors = to_categorical(fake_categories, self.num_classes+1)
    random_value_part = np.random.uniform(0,1,size=[16,100-(self.num_classes+1)])
    fake_values = np.concatenate((fake_vectors, random_value_part), axis=1)
    #fake_values = np.random.uniform(0,1,size=[16,100])
    images = self.generator.predict(fake_values)
    plt.figure(figsize=(10,10))

    for i in range(images.shape[0]):
      plt.subplot(4, 4, i+1)
      image = images[i, :, :, :]
      if self.img_channels == 1:
        image = np.reshape(image, [self.img_rows, self.img_cols])
      elif K.image_data_format() == 'channels_first':
        image = image.transpose(1,2,0)
      # implicit no need to transpose if channels are last
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

  def load_args(self):
    self.commandline_args = self.parser.parse_args()

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
