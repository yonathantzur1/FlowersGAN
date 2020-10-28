import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

NOISE_SIZE = 350
IMAGE_SIZE = 256

GENERATE_PATH = "./gan_flowers"
THRESHOLD = 0.5
AMOUNT = 5

GENERATOR_MODEL = "./models/generator.h5"
DISCRIMINATOR_MODEL = "./models/discriminator.h5"


def create_noise(images_amonut):
    return np.random.normal(0, 1, (images_amonut, NOISE_SIZE))


def generate_image(generator_model, discriminator_model):
    noise = np.random.normal(0, 1, (AMOUNT, NOISE_SIZE))
    generated_images = generator_model.predict(noise)
    predict = discriminator_model.predict(generated_images)

    while min(predict) < THRESHOLD:
        false_indexes = []
        for i in range(len(predict)):
            if predict[i] < THRESHOLD:
                false_indexes.append(i)
        noise = np.random.normal(0, 1, (len(false_indexes), NOISE_SIZE))
        fixed_images = generator_model.predict(noise)
        for i in range(len(false_indexes)):
            generated_images[false_indexes[i]] = fixed_images[i]
        predict = discriminator_model.predict(generated_images)

    print("Minimum accuracy: " + str(min(predict)[0]))
    print("Maximum accuracy: " + str(max(predict)[0]))
    generated_images = 0.5 * generated_images + 0.5
    for index in range(AMOUNT):
        image_array = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image_array[0:0 + IMAGE_SIZE, 0:0 + IMAGE_SIZE] = generated_images[index] * 255
        filename = os.path.join(GENERATE_PATH, str(index) + ".png")
        im = Image.fromarray(image_array)
        im.save(filename)


print("Generating...")
generate_image(load_model(GENERATOR_MODEL), load_model(DISCRIMINATOR_MODEL))
print("done...")
