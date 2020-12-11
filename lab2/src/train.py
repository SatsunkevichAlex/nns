__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""


import argparse
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential


LOG_DIR = 'logs'
SHUFFLE_BUFFER = 4
BATCH_SIZE = 256
NUM_CLASSES = 6
PARALLEL_CALLS=4
RESIZE_TO = 224
TRAINSET_SIZE = 14034
VALSET_SIZE = 3000


def display_image(log_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    glob_path = Path(current_dir + "/../food_tfr")
    file_list = [str(pp) for pp in glob_path.glob("*")]

    train_images = create_dataset(file_list, BATCH_SIZE)

    image = [np.reshape(i[:, :, :, :], (-1, 224, 224, 3)) for i in train_images.as_numpy_iterator()][0]
    l_channel = [np.reshape(i[:, :, :, 0], (-1, 224, 224, 1)) for i in train_images.as_numpy_iterator()][0]
    ab_channel = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in train_images.as_numpy_iterator()][0]

    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(log_dir)

    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        tf.summary.image("Image", image, step=0)
        tf.summary.image("Training data L channel", l_channel, step=0)
        tf.summary.image("Training data a channel", ab_channel, step=0)

    print("Images saved")


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    example = tf.io.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image']


def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .batch(batch_size)\
        .prefetch(batch_size)


def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    return model


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files')
    args.add_argument('--test', type=str, help='Glob pattern to collect test tfrecord files')

    log_dir = "C:/Users/Alex/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    display_image(log_dir)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_dir = Path(current_dir + "/../food_tfr")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]

    valid_dir = Path(current_dir + "/../food_test")
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]

    train_dataset = create_dataset(file_list_train, BATCH_SIZE)
    validation_dataset = create_dataset(file_list_valid, BATCH_SIZE)

    validation_y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in validation_dataset.as_numpy_iterator()][0]

    x = [np.reshape(i[:, :, :, 0], (-1, 224, 224, 1)) for i in train_dataset.as_numpy_iterator()][0]
    y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in train_dataset.as_numpy_iterator()][0]

    model = build_model()

    model.compile(
         optimizer=tf.optimizers.SGD(lr=0.01, momentum=0.9),
         loss=tf.keras.losses.mean_squared_error,
         metrics=[tf.keras.metrics.categorical_accuracy]
    )

    model.fit(
        x=x,
        y=y,
        epochs=1000,
        validation_data=validation_y.all(),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
        ]
    )

    # Test model
    output = model.predict(x)

    file_writer = tf.summary.create_file_writer(log_dir)
    # Output colorizations
    for i in range(3):
        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = x[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        with file_writer.as_default():
          tf.summary.image("{i}-img_result.png".format(i=i), np.reshape(cur, (1, 224, 224, 3)), step=4)

    print(model.summary())


if __name__ == '__main__':
    main()
