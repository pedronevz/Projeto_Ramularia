import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
from pathlib import Path

# Parâmetros
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Caminhos das pastas de treinamento e teste
path_train = Path('./Projeto_Ramularia/Healthy_Train50')
path_test_healthy = Path('./Projeto_Ramularia/Healthy_Test50')
path_test_diseased = Path('./Projeto_Ramularia/Disease_Test100')

# Funções de pré-processamento
def resize(image, height, width):
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

# Função de normalização
def normalize(image):
    return (image / 127.5) - 1

# Função de jitter aleatório
def random_jitter(image):
    image = tf.image.resize(image, [286, 286])
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image

# Função para carregar e pré-processar imagem
def load_and_preprocess_image(image_path, apply_jitter=False):
    # Lê e decodifica a imagem
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)

    # Converte a imagem para tf.float32
    image = tf.cast(image, tf.float32)

    # Redimensiona inicialmente para 256x256
    image = resize(image, IMG_HEIGHT, IMG_WIDTH)

    if apply_jitter:
        # Aplica jitter aleatório (para treinamento)
        image = random_jitter(image)

    # Normaliza a imagem
    image = normalize(image)

    return image

# Função de carregamento e pré-processamento para treinamento
def load_image_train(image_file):
    return load_and_preprocess_image(image_file, apply_jitter=True)

# Função de carregamento e pré-processamento para teste com etiqueta
def load_image_test(image_file, label):
    image = load_and_preprocess_image(image_file, apply_jitter=False)
    return image, label


# Criação do pipeline de entrada para treinamento
train_dataset = tf.data.Dataset.list_files(str(path_train / '*.jpg'))
train_dataset = train_dataset.map(lambda x: load_image_train(x), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Criação do pipeline de entrada para teste (dois diretórios com etiquetas)
test_dataset_healthy = tf.data.Dataset.list_files(str(path_test_healthy / '*.jpg'))
test_dataset_diseased = tf.data.Dataset.list_files(str(path_test_diseased / '*.jpg'))

test_dataset_healthy = test_dataset_healthy.map(lambda x: load_image_test(x, 0), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_diseased = test_dataset_diseased.map(lambda x: load_image_test(x, 1), num_parallel_calls=tf.data.AUTOTUNE)

# Concatenar os dois datasets de teste
test_dataset = test_dataset_healthy.concatenate(test_dataset_diseased)
test_dataset = test_dataset.batch(BATCH_SIZE)

# print(f"Number of images in test_dataset: {test_dataset.cardinality().numpy()}") # Checando se juntou corretamente os datasets de teste

"""
# Exibir algumas imagens de treinamento
for image_batch in train_dataset.take(1):
    plt.figure(figsize=(6, 6))
    batch_size = image_batch.shape[0]
    for i in range(min(4, batch_size)):
        plt.subplot(2, 2, i + 1)
        plt.imshow((image_batch[i] + 1) / 2)  # Converte a imagem de volta para o intervalo [0, 1] para exibição
        plt.axis('off')
    plt.show()
"""

# Exibir algumas imagens de teste com etiquetas
for image_batch, label_batch in test_dataset.take(1):
    plt.figure(figsize=(6, 6))
    batch_size = image_batch.shape[0]
    for i in range(batch_size):
        plt.subplot(2, 2, i + 1)
        plt.imshow((image_batch[i] + 1) / 2)  # Converte a imagem de volta para o intervalo [0, 1] para exibição
        plt.title('Healthy' if label_batch[i] == 0 else 'Diseased')
        plt.axis('off')
    plt.show()
