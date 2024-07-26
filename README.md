# Author: Fadi Badine
# Date created: 14/06/2020
# Last modified: 19/07/2023
# Description: Classify speakers using Fast Fourier Transform (FFT) and a 1D Convnet.

# Introduction
# This example demonstrates how to create a model to classify speakers from the frequency domain representation of speech recordings, obtained via Fast Fourier Transform (FFT).

# It shows the following:
# - How to use tf.data to load, preprocess and feed audio streams into a model
# - How to create a 1D convolutional network with residual connections for audio classification.

# Our process:
# - We prepare a dataset of speech samples from different speakers, with the speaker as label.
# - We add background noise to these samples to augment our data.
# - We take the FFT of these samples.
# - We train a 1D convnet to predict the correct speaker given a noisy FFT speech sample.

# Note:
# - This example should be run with TensorFlow 2.3 or higher, or tf-nightly.
# - The noise samples in the dataset need to be resampled to a sampling rate of 16000 Hz before using the code in this example. In order to do this, you will need to have installed ffmpg.

# Setup
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import shutil
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from IPython.display import display, Audio

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/
# and save it to ./speaker-recognition-dataset.zip
# then unzip it to ./16000_pcm_speeches
!kaggle datasets download -d avishkar001/iemocap-team-noob
!unzip -qq iemocap-team-noob.zip

DATASET_ROOT = "IEMOCAP_Team_Noob"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 1  # For a real training run, use EPOCHS = 100

# Data preparation
# The dataset is composed of 7 folders, divided into 2 groups:
# Speech samples, with 5 folders for 5 different speakers. Each folder contains 1500 audio files, each 1 second long and sampled at 16000 Hz.
# Background noise samples, with 2 folders and a total of 6 files. These files are longer than the speech samples, sampled at 16000 Hz.

# Let's sort the folders into the two groups, and split the speech samples into training and validation sets.
speech_samples = []
noise_samples = []
for root, dirs, files in os.walk(DATASET_AUDIO_PATH):
    for file in files:
        if file.endswith(".wav"):
            speech_samples.append(os.path.join(root, file))
for root, dirs, files in os.walk(DATASET_NOISE_PATH):
    for file in files:
        if file.endswith(".wav"):
            noise_samples.append(os.path.join(root, file))

# Split the speech samples into training and validation sets
total_samples = len(speech_samples)
num_val_samples = int(VALID_SPLIT * total_samples)

np.random.seed(SHUFFLE_SEED)
np.random.shuffle(speech_samples)

train_speech_samples = speech_samples[:-num_val_samples]
val_speech_samples = speech_samples[-num_val_samples:]

# Function to decode the wav files
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

# Function to get the label from the filepath
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

# Function to preprocess the audio files
def preprocess(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio = decode_audio(audio_binary)
    label = get_label(file_path)
    return audio, label

# Create TensorFlow datasets from the file paths
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(train_speech_samples)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val_speech_samples)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Function to add noise to audio samples
def add_noise(audio, noise, scale=SCALE):
    prop = tf.math.reduce_std(audio) / tf.math.reduce_std(noise)
    noisy_audio = audio + noise * prop * scale
    return noisy_audio

# Augment the data by adding noise
noises = []
for noise_sample in noise_samples:
    noise_audio_binary = tf.io.read_file(noise_sample)
    noise_audio = decode_audio(noise_audio_binary)
    noises.append(noise_audio)

noises_ds = tf.data.Dataset.from_tensor_slices(noises)
noises_ds = noises_ds.shuffle(buffer_size=len(noises)).repeat()

def add_noise_to_dataset(audio, label):
    noise_audio = next(iter(noises_ds))
    noisy_audio = add_noise(audio, noise_audio)
    return noisy_audio, label

train_ds = train_ds.map(add_noise_to_dataset, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(add_noise_to_dataset, num_parallel_calls=AUTOTUNE)

# Batch the data
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# Prefetch the data for optimal performance
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Model building
def residual_block(x, filters, conv_num=3, activation="relu"):
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for _ in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPooling1D(pool_size=2)(x)

def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Conv1D(16, 3, padding="same")(inputs)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 2)
    x = residual_block(x, 128, 2)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, x)

# Compile the model
model = build_model((SAMPLING_RATE, 1), len(set(train_ds.map(lambda audio, label: label))))
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Train the model
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# Evaluate the model
loss, acc = model.evaluate(val_ds)
print(f"Validation accuracy: {acc:.3f}")

# Save the model
model.save("speaker_recognition_model")
