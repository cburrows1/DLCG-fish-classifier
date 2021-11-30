# See https://keras.io/guides/transfer_learning/

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1.5GB of memory on the GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1536)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



dataset_path = "dataset"
SEED = 14322

BATCH_SIZE = 16
IMG_SIZE = (150, 150)

dataset_dir = os.path.abspath(dataset_path)
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  shuffle=True,
  label_mode='binary',
  color_mode='rgb',
  validation_split=0.2,
  subset="training",
  seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  shuffle=True,
  label_mode='binary',
  color_mode='rgb',
  validation_split=0.2,
  subset="validation",
  seed=SEED
)


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
def pltConfusionMatrix(results, expected):
    conf_matrix = confusion_matrix(results, expected)
    norm_conf_matrix = np.array([normalize(x, axis=1, norm='l1') for x in conf_matrix])

    title = "Fish Classifier Confusion Matrix"
    _, ax = plt.subplots()
    afig = ax.matshow(norm_conf_matrix,cmap=plt.cm.RdYlGn)
    plt.colorbar(afig)
    ax.set_title(title)
    for i in range(len(norm_conf_matrix)):
        for j in range (len(norm_conf_matrix[0])):
            c = round(norm_conf_matrix[j][i],3)
            ax.text(i, j, str(c), va='center', ha='center')
    plt.savefig("conf_matrix.png")

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
plt.savefig("input-aug.png")

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0].numpy().astype("int32"))
    plt.title(int(label[0]))
    plt.axis("off")
plt.savefig("input-imgs.png")

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

epochs = 2#0
model.fit(train_ds, epochs=epochs, validation_data=validation_ds )

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

epochs = 0#10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


test_path = "testset"
test_dir = os.path.abspath(test_path)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='binary',
    color_mode='rgb'
)

results = model.evaluate(test_ds)
print("test loss, test acc:", results)

predictions = model.predict(test_ds).argmax(axis=1) #this is all zero this is the issue

expected = np.concatenate([y for x, y in test_ds], axis=0).T[0]
print(len(predictions), predictions[:100])
print(len(expected), expected[:100])
pltConfusionMatrix(predictions,expected)