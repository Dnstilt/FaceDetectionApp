# Custom L1 Distance layer module
#Load de o custom model

#Importando as dependencias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

#Codigo do  Custom L1 Distance Layer
class L1Dist(Layer):

    #Metodo init - Heranca
    def __init__(self, **kwargs):
        super().__init__()

    #Aqui e feita a magia
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)