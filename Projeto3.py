import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import structural_similarity as ssim
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# Parâmetros
BUFFER_SIZE = 400
BATCH_SIZE = 8
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Caminhos das pastas de treinamento e teste
path_train = Path('./Projeto_Ramularia/Healthy_Train50')
path_test_healthy = Path('./Projeto_Ramularia/Healthy_Test50')
path_test_diseased = Path('./Projeto_Ramularia/Disease_Test100')

# Funções de pré-processamento
def resize(image, height, width):
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def random_jitter(image):
    image = tf.image.resize(image, [286, 286])
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image

def normalize(image):
    return (image / 127.5) - 1

def load_and_preprocess_image(image_path, apply_jitter=False):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = resize(image, IMG_HEIGHT, IMG_WIDTH)
    if apply_jitter:
        image = random_jitter(image)
    image = normalize(image)
    return image

def load_image_train(image_file):
    image = load_and_preprocess_image(image_file, apply_jitter=True)
    return image, image

train_dataset = tf.data.Dataset.list_files(str(path_train / '*.jpg'))
train_dataset = train_dataset.map(lambda x: load_image_train(x), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_test(image_file, label):
    image = load_and_preprocess_image(image_file, apply_jitter=False)
    return image, label

test_dataset_healthy = tf.data.Dataset.list_files(str(path_test_healthy / '*.jpg'))
test_dataset_diseased = tf.data.Dataset.list_files(str(path_test_diseased / '*.jpg'))

test_dataset_healthy = test_dataset_healthy.map(lambda x: load_image_test(x, 0), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_diseased = test_dataset_diseased.map(lambda x: load_image_test(x, 1), num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset_healthy.concatenate(test_dataset_diseased)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result

def upsample(filters, size, dropout=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', 
                                         kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(256, 4, dropout=True),
        upsample(128, 4),
        upsample(64, 4),
    ]

    last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                           activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def discriminator_model():
    initializer = tf.random_normal_initializer(0., 0.02)
    input_img = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    target_img = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')
    
    x = tf.keras.layers.Concatenate()([input_img, target_img])

    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=[input_img, target_img], outputs=x)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(generated_output, target, gen_output):
    l1_loss = tf.keras.losses.MeanAbsoluteError()(target, gen_output)
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated_output), generated_output)
    total_gen_loss = gan_loss + LAMBDA * l1_loss
    return total_gen_loss

@tf.function
def train_step(input_image, target, generator, discriminator, gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, target, gen_output)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

LAMBDA = 100

generator = unet_model(output_channels=3)
discriminator = discriminator_model()

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
early_stopping_patience = 5

best_gen_loss = float('inf')
epochs_since_last_improvement = 0

for epoch in range(EPOCHS):
    start = time.time()

    total_gen_loss = 0
    total_disc_loss = 0

    for input_image, target in train_dataset:
        gen_loss, disc_loss = train_step(input_image, target, generator, discriminator, gen_optimizer, disc_optimizer)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss

    avg_gen_loss = total_gen_loss / len(train_dataset)
    avg_disc_loss = total_disc_loss / len(train_dataset)

    print(f'Epoch {epoch+1}/{EPOCHS}, Generator Loss: {avg_gen_loss.numpy()}, Discriminator Loss: {avg_disc_loss.numpy()}')

    if avg_gen_loss < best_gen_loss:
        best_gen_loss = avg_gen_loss
        epochs_since_last_improvement = 0
        checkpoint.save(file_prefix=checkpoint_prefix)
    else:
        epochs_since_last_improvement += 1
        if epochs_since_last_improvement >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

    print(f'Tempo de Epoch {epoch+1}: {time.time() - start} segundos')

def display_sample_image(input_image, generated_image):
    plt.figure(figsize=(12, 6))

    # Display input image
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow((input_image + 1) / 2)  # Reverter normalização para exibição
    plt.axis('off')

    # Display generated image
    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    plt.imshow((generated_image + 1) / 2)  # Reverter normalização para exibição
    plt.axis('off')

    plt.show()

def rgb_to_lab(rgb_image):
    # RGB para LAB
    return rgb2lab(rgb_image)

def calculate_ciede2000_anomaly_score(original_image, generated_image):
    # Converter as imagens RGB para LAB
    original_lab = rgb_to_lab(original_image)
    generated_lab = rgb_to_lab(generated_image)
    
    # Flatten as imagens para calcular o CIEDE2000 para cada pixel
    original_lab_flat = original_lab.reshape(-1, 3)
    generated_lab_flat = generated_lab.reshape(-1, 3)
    
    # Calcula a diferença de cor para cada pixel
    total_delta_e = 0
    num_pixels = original_lab_flat.shape[0]
    
    for i in range(num_pixels):
        original_pixel = LabColor(*original_lab_flat[i])
        generated_pixel = LabColor(*generated_lab_flat[i])
        delta_e = delta_e_cie2000(original_pixel, generated_pixel)
        total_delta_e += delta_e
    
    # Calcula a pontuação média de anomalia
    average_delta_e = total_delta_e / num_pixels
    return average_delta_e

# Testar o modelo com uma imagem do dataset de teste
for input_image, target in test_dataset.take(1):
    prediction = generator(input_image, training=False)
    display_sample_image(input_image[0], prediction[0])
    
    # Testar o cálculo da pontuação de anomalia
    input_image_np = (input_image[0] + 1) / 2  # Reverter normalização
    prediction_np = (prediction[0] + 1) / 2  # Reverter normalização
    anomaly_score = calculate_ciede2000_anomaly_score(input_image_np.numpy(), prediction_np.numpy())
    print(f'Anomaly Score: {anomaly_score}')

generator.save('pix2pix_generator.h5')
discriminator.save('pix2pix_discriminator.h5')

def load_image_from_path(image_path, apply_jitter=False):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = resize(image, IMG_HEIGHT, IMG_WIDTH)
    if apply_jitter:
        image = random_jitter(image)
    image = normalize(image)
    return tf.expand_dims(image, 0)  # Adiciona uma dimensão para o batch

# Caminho da imagem específica para teste
specific_image_path = 'Projeto_Ramularia/Disease_Test100/a984-987 ad_1.jpg'

# Carregar e pré-processar a imagem
input_image = load_image_from_path(specific_image_path, apply_jitter=False)

# Gerar a imagem a partir do gerador
generated_image = generator(input_image, training=False)

# Exibir a imagem original e a imagem gerada
display_sample_image(input_image[0], generated_image[0])
