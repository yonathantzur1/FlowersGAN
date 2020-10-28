import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, \
    ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

BASE_PATH = "/content/drive/My Drive/flowers/"
DATASET_PATH = "./dataset256/"

GENERATOR_MODEL = BASE_PATH + "models/generator.h5"
DISCRIMINATOR_MODEL = BASE_PATH + "models/discriminator.h5"
# GENERATOR_MODEL = None
# DISCRIMINATOR_MODEL = None


# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4

# Size vector to generate images from
NOISE_SIZE = 512

# Configuration
EPOCHS = 500
BATCH_SIZE = 128
DISCRIMINATOR_OPTIMIZER = Adam(2e-4, 0.5)
GENERATOR_OPTIMIZER = Adam(2e-4, 0.5)
SAVE_FREQ = 3

IMAGE_SIZE = 256  # rows/cols
IMAGE_CHANNELS = 3


def build_discriminator(image_shape):
    if DISCRIMINATOR_MODEL is not None:
        return load_model(DISCRIMINATOR_MODEL)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    input_image = Input(shape=image_shape)
    validity = model(input_image)

    print("------------discriminator------------")
    model.summary()
    return Model(input_image, validity)


def build_generator(noise_size, channels):
    if GENERATOR_MODEL is not None:
        return load_model(GENERATOR_MODEL)

    model = Sequential()
    model.add(Dense(4 * 4 * 1024, activation="relu", input_dim=noise_size))
    model.add(Reshape((4, 4, 1024)))

    model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(channels, kernel_size=5, strides=2, padding="same"))
    model.add(Activation("tanh"))
    input = Input(shape=(noise_size,))
    generated_image = model(input)

    print("------------generator------------")
    model.summary()
    return Model(input, generated_image)


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def get_time():
    return int(round(time.time() * 1000))


def save_images(noise, name="flowers"):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3), 255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    output_path = BASE_PATH + "output"
    create_dir(output_path)
    filename = os.path.join(output_path, name + ".png")
    im = Image.fromarray(image_array)
    im.save(filename)


def save_models(generator_model, discriminator_model,
                generator_name="generator.h5", discriminator_name="discriminator.h5"):
    output_path = BASE_PATH + "models/"
    create_dir(output_path)
    generator_model.save(output_path + generator_name)
    discriminator_model.save(output_path + discriminator_name)


# train #
image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
discriminator = build_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=DISCRIMINATOR_OPTIMIZER, metrics=["accuracy"])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer=GENERATOR_OPTIMIZER, metrics=["accuracy"])
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))

for epoch in range(EPOCHS):
    print("epoch:" + str(epoch + 1))
    all_images = []
    for filename in os.listdir(DATASET_PATH):
        all_images.append(os.path.join(DATASET_PATH, filename))
    random.shuffle(all_images)

    max_batch = math.ceil(len(all_images) / BATCH_SIZE)

    for batch in range(max_batch):
        idx = []

        while len(idx) < BATCH_SIZE and len(all_images) > 0:
            idx.append(all_images[0])
            all_images = all_images[1:]

        batch_size = min(BATCH_SIZE, len(idx))

        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        batch_images = np.array([np.array(Image.open(fname)) for fname in idx])

        data_augmentation = ImageDataGenerator(rotation_range=10,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               shear_range=0.01,
                                               zoom_range=[0.8, 1.2],
                                               horizontal_flip=True,
                                               vertical_flip=False,
                                               fill_mode='reflect',
                                               data_format='channels_last',
                                               brightness_range=[0.8, 1.2],
                                               dtype=str(batch_images.dtype))

        batch_augmentation = data_augmentation.flow(batch_images, batch_size=batch_size)[0]
        np.reshape(batch_augmentation, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
        x_real = batch_augmentation / 127.5 - 1

        # np.reshape(batch_images, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
        # x_real = batch_images / 127.5 - 1

        batch_noise = np.random.normal(0, 1, (batch_size, NOISE_SIZE))
        x_fake = generator.predict(batch_noise)

        discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
        discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

        discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
        generator_metric = combined.train_on_batch(batch_noise, y_real)

    gen_acc = float("%.3f" % (100 * generator_metric[1]))
    gen_loss = float("%.3f" % generator_metric[0])
    disc_acc = float("%.3f" % (100 * discriminator_metric[1]))
    disc_loss = float("%.3f" % discriminator_metric[0])

    print("Generator loss: " + str(gen_loss) + ", Generator accuracy: " + str(gen_acc))
    print("Discriminator loss: " + str(disc_loss) + ", Discriminator accuracy: " +
          str(disc_acc))
    print("-----------------------------------------------------------------")

    if gen_acc > 35:  # (epoch + 1) % SAVE_FREQ == 0:
        print("saving models...")
        print()
        save_images(fixed_noise, str(get_time()))
        save_models(generator, discriminator)
