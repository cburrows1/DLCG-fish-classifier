# See https://keras.io/guides/transfer_learning/

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os



_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path = tf.keras.utils.get_file("cats_and_dogs.zip", _URL, extract=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator()

cvd_ds = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")


BATCH_SIZE = 32
IMG_SIZE = (150, 150)

train_dir = os.path.join(cvd_ds, 'train')
validation_dir = os.path.join(cvd_ds, 'validation')

def image_flow(path):
    return img_gen.flow_from_directory(path,
                                target_size=IMG_SIZE,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                class_mode='binary',
                                color_mode='rgb'
                                )

train_ds = tf.data.Dataset.from_generator(
    lambda: image_flow(train_dir), 
    output_signature=(
        tf.TensorSpec(shape=(32,150,150,3), dtype=tf.int32),
        tf.TensorSpec(shape=(32,), dtype=tf.int32))
)

validation_ds = tf.data.Dataset.from_generator(
    lambda: image_flow(validation_dir),
    output_signature=(
        tf.TensorSpec(shape=(32,150,150,3), dtype=tf.int32),
        tf.TensorSpec(shape=(32,), dtype=tf.int32))
)


data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)


for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")
plt.show()


base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)