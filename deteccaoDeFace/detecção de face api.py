#Use o Jupter ou o colab e vá rodando os blocos de código um por vez. Troque o caminho dos arquivos para os do seu uso.
#Importando
import cv2 #Opencv
import os #Acessar e manipular diretórios e arquivos 
import random
import numpy as np #Para trabalhar com diferentes tipos de array
from matplotlib import pyplot as plt #Plotando visualizações de dados


# Importando tensorflow dependencies - Functional API
from keras.models import Model 
#Model(inputs=[inputimage, verificationimage], outputs =[1, 0])
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
#Os modelos precisam de diferentes camadas e para isso servem os imports
import tensorflow as tf


#Crie uma pasta data no mesmo diretório do projeto e coloque o caminho no 'data'
#Criando caminhos para os diretórios 
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


#Criando os diretórios
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


#Coletando os dados http://vis-www.cs.umass.edu/lfw/
#Descomprimindo dados
#Deve-se usar o http://vis-www.cs.umass.edu/lfw/lfw.tgz para baixar o arquivo de fotos.
#Descomprimi o arquivo
!tar -xf lfw.tgz
#Movendo os arquivos de imagens de download para o negative. 
for directory in os.listdir('lfw'): 
   for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

#Coletando as positive e anchor fotos.
#biblioteca que gera identificadores únicos para cada imagem
import uuid


os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

#Esta parte vai captar as imagems para treino do modelo.
#Conectando com a webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    
    ret, frame = cap.read()
    #Definindo tamnho do frame 250x250pxq
    frame = frame[120:120+250, 200:200+250, :]
    
    #Coletando anchors
    #Seta tecla para coleta
    if cv2.waitKey(1) & 0XFF == ord('a'):
        #Criando caminho único do arquivo   
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Gravando anchor image
        cv2.imwrite(imgname, frame)
        
    # Coletando positives
    #Seta tecla para coleta
    if cv2.waitKey(1) & 0XFF == ord('p'):q
        #Criando caminho único do arquivo    
    imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Gravando positive image
    cv2.imwrite(imgname, frame)   
     
    #Mostrando imagem
    cv2.imshow('Image Collection', frame)
    
    #Setando tecla para finalizarp
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

#Soltando a webcam
cap.release()
#Fechando a janela da imagem
cv2.destroyAllWindows()        
        

#Após capturar as imagens essa função trabalha as imagens para gerar variantes.
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2)) #modifica o brilho 
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3)) #Modifica o contraste
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100))) #Espelha a imagem
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100))) #modifica a qualidade
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100))) #Muda a saturação 
            
        data.append(img)
    
    return data

import os
import uuid
import numpy as np
import tensorflow as tf

#Ampliando o número de dados
#Deve-se repetir o for com POS_PATH
for file_name in os.listdir(os.path.join(ANC_PATH)):
    img_path = os.path.join(ANC_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        
        cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


#Lendo e preprocessando imagens


#Padroniza as imagens em .jpg e seleciona x números para serem usados 
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(1500)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(1500)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(1500)

#Teste
dir_test = anchor.as_numpy_iterator()
print(dir_test.next())


#Preprocessando as imagens
def preprocess(file_path):
    
    #Lendo imagem do file path
    byte_img = tf.io.read_file(file_path)
    # Load imagem em bytes
    img = tf.io.decode_jpeg(byte_img)
    
    #Transformando a imagem 100x100x3
    img = tf.image.resize(img, (105,105))
    
    #Colocando valor entre 0 e 1 
    img = img / 255.0

    # Return image
    return img

dataset.map(preprocess)


#Criando dataset nomeado 
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)



#Construindo o treino e partição do treino
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)
 
#Teste
data

#construindo dataloader pipeline

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

data

#Treinando partições
train_data = data.take(round(len(data)*.7))#Pega uma porcentagem da partição 
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

#Teste
train_data

#Testando partições
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


#Modelo de engenharia
#Construindo as camadas de embeddig deeplearning
inp = Input(shape=(105,105,3), name='input_image')

c1 = Conv2D(64, (10,10), activation='relu')(inp)

m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)

mod= Model(inputs=[inp], outputs=[d1], name='embedding')

#Teste
mod.summary()

def make_embedding(): 
    inp = Input(shape=(105,105,3), name='input_image')
    
    #Primeiro bloco
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    #Segundo bloco
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    #Terceiro bloco 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    #Bloco final embedding
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()

#Teste
embedding

#Construindo a camada distance
#Siamese L1 Distance class
class L1Dist(Layer):
    
    #Método init - herança 
    def __init__(self, **kwargs):
        super().__init__()
       
    #Aqui acontece a mágica - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)#valor absoluto

l1 = L1Dist()


#Construindo o modelo Siamese
input_image = Input(name='input_img', shape=(105,105,3))
validation_image = Input(name='validation_img', shape=(105,105,3))


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


siamese_layer = L1Dist()

distances = siamese_layer(inp_embedding, val_embedding)

classifier = Dense(1, activation='sigmoid')(distances)

classifier

siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_network.summary()

def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()

#Treinando
#Setup loss e Optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

#Estabelecendo os checkpoints
#Diretório para salvar os checkpoints
checkpoint_dir = './training_checkpoints'
#Estabekecendo o prefixo e o caminho do arquivo
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
#Salvando o optimizer e o model
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

#Costruiondo o Train Step Funcion
#Faz a predição, calcula a perda, deriva os gradientes
# e calcula os novos pesos e aplica 

test_batch = train_data.as_numpy_iterator()

batch_1 = test_batch.next()

X = batch_1[:2]

y = batch_1[2]

y

#Começamos a construir uma rede neural aqui
@tf.function #compila o que está na função 
def train_step(batch):
    
    #Grava todas as operações  
    with tf.GradientTape() as tape:     
        #Pega as anchor e positive/negative image
        X = batch[:2]
        #Pega o label
        y = batch[2]
        
        #Passa dados ao modelo para fazer a predição 
        yhat = siamese_model(X, training=True)
        #Calcula as perdas
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    #Calcula os gradientes
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # Usa o optmizador Adam para calcular
    #Calcula e renova os pesos e aplica o siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    #REtorna as perdas
    return loss
 
#Construindo o Training Loop
# Import metric calculations
from keras.metrics import Precision, Recall

#Enquanto o train spet foca em um único batch(lote) o o loop aqui 
#itera sobre todo o data set
def train(data, EPOCHS):
    # Loop entre os epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        #Cria o objeto da metrica 
        r = Recall()
        p = Precision()
        
        #Loop em cada lote
        for idx, batch in enumerate(data):
            #Roda a função Train_step
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        #Salva os checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50

train(train_data, EPOCHS)

#Avaliando o modelo
#Importando os calculos da metrica
from keras.metrics import Precision, Recall

#Pegando umlote de dados
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])

#Pos processamento dos resultados
[1 if prediction > 0.5 else 0 for prediction in y_hat ] #listcomprehensiom

y_true

#Calculando as metricas
#Criando objeto métrico  
m = Recall()

#Calculando recall value 
m.update_state(y_true, y_hat)

#Retorno do Recall Result
m.result().numpy()

# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())

#Visualizando dados
#Setando tamanho da figura
plt.figure(figsize=(10,8))

#Setando primeiro subplot subplot function renderiza os plots.
plt.subplot(1,2,1)
plt.imshow(test_input[2])

#Setando segundo subplot 
plt.subplot(1,2,2)
plt.imshow(test_val[3])

#Renderiza de forma mais limpa
plt.show()

#Salvando o modelo

# Save weights
siamese_model.save('siamesemodelv2.h5')

L1Dist

# Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])

# View model summary
siamese_model.summary()

#Real time teste
#Verificacao
#application_data\verification_images

os.listdir(os.path.join('application_data', 'verification_images'))

os.path.join('application_data', 'input_image', 'input_image.jpg')

for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)

def verify(model, detection_threshold, verification_threshold):
    #Construindo array de resultados
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        #Fazendo predições 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    #Detection Threshold: Medida pela qual será  aceita a verificação 
    detection = np.sum(np.array(results) > detection_threshold)
    
    #Verification Threshold: Proporção de predições positivas/ Total de exemplos positivos 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    #Positiva ou não a verificação 
    verified = verification > verification_threshold
    
    return results, verified

#Verificação Real time com OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    #Definindo tamnho do frame 250x250pxq
    qframe = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         h, s, v = cv2.split(hsv)

         lim = 255 - 10
         v[v > lim] = 255
         v[v <= lim] -= 10
        
         final_hsv = cv2.merge((h, s, v))
         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.9, 0.7)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
