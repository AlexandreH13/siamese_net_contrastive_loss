import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers

from utils import euclidean_distance

def build_siamese() -> keras.models.Model:

    '''
    Cria arquitetura siamesa. Duas camadas de convolução e saída sendo distância euclidiana
    '''

    input = layers.Input((28, 28, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(4, (5, 5), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="tanh")(x)
    embedding_network = keras.Model(input, x)


    input_1 = layers.Input((28, 28, 1))
    input_2 = layers.Input((28, 28, 1))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    siamese = keras.Model(inputs=[input_1, input_2], outputs=merge_layer)

    return siamese

def contrastive_loss(y, preds, margin=1):
	'''
    Implementa a contrastive loss
    '''
    
	y = tf.cast(y, preds.dtype) # Cast necessário para não termos tipos diferentes

	ls = K.square(preds) # Para pares similares
	ld = K.square(K.maximum(margin - preds, 0)) # Para pares diferentes
	loss = K.mean(y * ls + (1 - y) * ld)
	
	return loss