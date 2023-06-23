# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from faceApp.layers import L1Dist
import os
import numpy as np


# Build app and layout
class CamApp(App):

    def build(self):
        #Componentes centrais do layout
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verificar", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verificação Iniciada", size_hint=(1, .1))

        #Adicionando itens ao layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #importando tensorflow/keras model
        self.model = tf.keras.models.load_model('C:\\Users\\Denis\\app\\siamesemodelv2.h5.h5', custom_objects={'L1Dist': L1Dist})

        #Trazendo o vídeo da webcan
        self.capture = cv2.VideoCapture(0)
        #Roda a funçáo update para continuar a rodar os frames
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    #continua rodando a webcam no App
    def update(self, *args):
        #Lendo o frame do opencv
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :] #Tamanho e canais

        #Virando na horizontal e convertendo imagem para textura
        buf = cv2.flip(frame, 0).tobytes()
        #Começando a converter imagem a uma textura para renderizar no App
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    #Rodando a imagem do diretório e convertendo em 105 x 105
    def preprocess(self, file_path):
        #Lendo arquivo do arquivo
        byte_img = tf.io.read_file(file_path)
        #trazendo a imgem
        img = tf.io.decode_jpeg(byte_img)

        #passos do processamento redimencionando a imagem
        img = tf.image.resize(img, (105, 105))
        #Colocando a imagem com o valor entre 0 e 1
        img = img / 255.0

        #Retorna a imagem
        return img

    #Função de varificação da pessoa
    def verify(self, *args):
        #Especifica qual o nível de aceitação do programa
        detection_threshold = 0.8 #Porcentagem mínima de aceitação
        verification_threshold = 0.6 #Proporção da predição precisa ser positiva para aceitar

        #Captura o frame da webcam
        SAVE_PATH = os.path.join('C:\\Users\\Denis\\app\\application_data', 'C:\\Users\\Denis\\app\\application_data\\input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        #Construindo o array de resultados Build
        results = []
        for image in os.listdir(os.path.join('C:\\Users\\Denis\\app\\application_data', 'C:\\Users\\Denis\\app\\application_data\\verification_images')):
            input_img = self.preprocess(os.path.join('C:\\Users\\Denis\\app\\application_data', 'C:\\Users\\Denis\\app\\application_data\\input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('C:\\Users\\Denis\\app\\application_data', 'verification_images', image))

            #Faz a predição
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        #Valor da predição para validar ou não
        detection = np.sum(np.array(results) > detection_threshold)

        #Valor da verificação, proporção de predições positivas/total de amostras positivas
        verification = detection / len(os.listdir(os.path.join('C:\\Users\\Denis\\app\\application_data', 'verification_images')))
        verified = verification > verification_threshold

        #Setando o texto da verificação
        self.verification_label.text = 'Verificado' if verified == True else 'Não verificado'

        #Detalhes da saída
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.6))

        return results, verified


if __name__ == '__main__':
    CamApp().run()