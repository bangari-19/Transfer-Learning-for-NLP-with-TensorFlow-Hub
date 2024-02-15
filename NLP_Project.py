import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
from IPython import display

import pathlib
import shutil
import tempfile

# Install TensorFlow Docs for additional utilities
!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

# Check TensorFlow and TensorFlow Hub versions, and GPU availability
print("Version: ", tf.__version__)
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Create directory for TensorBoard logs
logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# Load data: decompress file and read into pandas DataFrame
df = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip', compression='zip', low_memory=False)
df.shape

from sklearn.model_selection import train_test_split

# Split datasets into training and validation subsets, controlling randomness with seed value and ensuring class distribution balance
train_df, remaining = train_test_split(df, random_state=42, train_size=0.01, stratify=df.target.values)
valid_df, _ = train_test_split(remaining, random_state=42, train_size=0.001, stratify=remaining.target.values)
train_df.shape, valid_df.shape

def train_and_evaluate_model(module_url, embed_size, name, trainable=False):
  # Load pre-trained model from TensorFlow Hub
  hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype=tf.string, trainable=trainable)
  # Create custom model for text classification
  model = tf.keras.models.Sequential([
                                      hub_layer,
                                      tf.keras.layers.Dense(256, activation='relu'),  # Dense layer with 256 neurons, using ReLU activation function
                                      tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  # Compile model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.losses.BinaryCrossentropy(),
                metrics=[tf.metrics.BinaryAccuracy(name='accuracy')])

  model.summary()

  # Fit model to our data
  history = model.fit(train_df['question_text'], train_df['target'],
                      epochs=100,
                      batch_size=32,
                      validation_data=(valid_df['question_text'], valid_df['target']),
                      callbacks=[tfdocs.modeling.EpochDots(),
                                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min'),
                                 tf.keras.callbacks.TensorBoard(logdir/name)],
                      verbose=0)
  return history

module_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

histories = {}
# Store model names as keys in the histories dictionary
histories['universal-sentence-encoder-large'] = train_and_evaluate_model(module_url,
                                                                     embed_size=512,
                                                                     name='universal-sentence-encoder-large')
                                                                     
# Plot accuracy curves for models
plt.rcParams['figure.figsize'] = (12, 8)
plotter = tfdocs.plots.HistoryPlotter(metric='accuracy')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Accuracy Curves for Models")
plt.show()

# Plot loss curves for models
plotter = tfdocs.plots.HistoryPlotter(metric='loss')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Loss Curves for Models")
plt.show()

# Load the TensorBoard extension
%load_ext tensorboard
# Visualize the training process with TensorBoard
%tensorboard --logdir {logdir}
