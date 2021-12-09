# See https://keras.io/guides/transfer_learning/

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

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

class PlotLearning(keras.callbacks.Callback):
  def __init__(self):
    self.metrics = {}

  def on_epoch_end(self, epoch, logs={}):
    for metric in logs:
      if metric in self.metrics:
        self.metrics[metric].append(logs[metric])
      else:
        self.metrics[metric] = [logs[metric]]

    f, axs = plt.subplots(1, len(logs), figsize=(15,5))
    for i, metric in enumerate(logs):
      axs[i].plot(range(1, epoch + 2),
          self.metrics[metric], 
          label=metric)
              
      axs[i].legend()
      axs[i].grid()
      axs[i].set_xlabel("Epoch number")
      axs[i].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig("train_graph.png")


SEED = 14322

BATCH_SIZE = 16
IMG_SIZE = (150, 150)

natives_path = "dataset/natives"
nonnatives_path = "dataset/nonnatives"

natives_dir = os.path.abspath(natives_path)
num_native = len(os.listdir(natives_dir+"/images"))
train_ds_natives = tf.keras.utils.image_dataset_from_directory(
  natives_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  labels=[0]*num_native,
  shuffle=True,
  label_mode='int',
  color_mode='rgb',
  seed=SEED

)

nonnatives_dir = os.path.abspath(nonnatives_path)
num_nonnative = len(os.listdir(nonnatives_dir+"/images"))
train_ds_nonnatives = tf.keras.utils.image_dataset_from_directory(
  nonnatives_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  labels=[1]*num_nonnative,
  shuffle=True,
  label_mode='int',
  color_mode='rgb',
  seed=SEED
)

resampled_ds = tf.data.experimental.sample_from_datasets([train_ds_natives.unbatch(), train_ds_nonnatives.unbatch().repeat(num_native // num_nonnative)], weights=[0.5, .5])
train_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomCrop(100, 100),
  ]
    
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


base_model = keras.applications.Xception(
    weights="imagenet",
    input_shape=(150, 150, 3),
    include_top=False,
)

base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs,training=True)
resize_layer = layers.Resizing(150,150)
x = resize_layer(x)

scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
x = keras.layers.Dense(2048, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(2048, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(2048, activation='relu')(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20

callbacks_list = [PlotLearning()]

model.fit(train_ds, epochs=epochs, callbacks=callbacks_list)

base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 5
model.fit(train_ds, epochs=epochs)


natives_path = "testset/natives"
nonnatives_path = "testset/nonnatives"

natives_dir = os.path.abspath(natives_path)
num_native = len(os.listdir(natives_dir+"/images"))
test_ds_natives = tf.keras.utils.image_dataset_from_directory(
  natives_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  labels=[0]*num_native,
  shuffle=False,
  label_mode='int',
  color_mode='rgb',
  seed=SEED
)

nonnatives_dir = os.path.abspath(nonnatives_path)
num_nonnative = len(os.listdir(nonnatives_dir+"/images"))
test_ds_nonnatives = tf.keras.utils.image_dataset_from_directory(
  nonnatives_dir,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  labels=[1]*num_nonnative,
  shuffle=False,
  label_mode='int',
  color_mode='rgb',
  seed=SEED
)
total = num_native + num_nonnative
resampled_ds = tf.data.experimental.sample_from_datasets([test_ds_natives.unbatch(), test_ds_nonnatives.unbatch().repeat(num_native // num_nonnative)], weights=[0.5, 0.5])
test_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)


results = model.evaluate(test_ds)
print("test loss, test acc:", results)

images = []
expected_labels = []
for x, y in test_ds.take(-1):
  images.append(x)
  expected_labels.append(y)

images = np.concatenate(images)
predictions = model.predict(images)
predictions = tf.nn.sigmoid(predictions).numpy().T[0]
predictions = tf.round(predictions)



def pltConfusionMatrix(results, expected):
    norm_conf_matrix = confusion_matrix(expected, results, labels=[0,1])#, normalize='true')

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


expected_labels = np.concatenate(expected_labels, axis=0)
pltConfusionMatrix(predictions,expected_labels)

