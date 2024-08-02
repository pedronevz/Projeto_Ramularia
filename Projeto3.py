import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
from pathlib import Path

dataset_name = "dados_treino"

path_train = Path('./Projeto_Ramularia/Healthy_Train50')

sample_image = tf.io.read_file(str(path_train / "leaf a24-a27 ab_2.jpg"))
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)

# Converte os valores de pixel para o intervalo [0, 1] e o tipo tf.float32
sample_image = tf.image.convert_image_dtype(sample_image, dtype=tf.float32)

# Exibe a imagem
plt.figure()
plt.imshow(sample_image)
plt.axis('off')  # Oculta os eixos
plt.show()