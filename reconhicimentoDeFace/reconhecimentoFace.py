!pip install opencv-python # instalando opencv
!pip install mediapipe # instalando mediapipe
import cv2
import mediapipe as mp

#inicializar opencv e medapipe
webcam = cv2.VideoCapture(0) # uso 0 pois só tenho uma webcan.
if webcam.isOpened():
  print("Conexão bem sucedida com webcam.")
# usa-se o mediapipe para se ter a marcação do rosto.
reconhe_rosto = mp.solutions.face_detection

# quando passar a imagem ele é quem vai detectar se tem rosto.
reconhecedor_de_rosto = reconhe_rosto.FaceDetection()

# vai desenhar os pontos e o quadrado do rosto.
desenho = mp.solutions.drawing_utils

while True:
  #Ler info da web. Recebe dois argumentos,
  #primeiro verifica se tem infos vindos da webcam V ou F
  #segundo caso tenha info joga dentro da variável a imagem capturada.
  verificador, frame = webcam.read()
  if not verificador:
    break
  #reconhecer rosto.


  lista_rostos = reconhecedor_de_rosto.process(frame)
  #Se tiver algum rosto na lista, para cada rosto desenhe o rosto.
  if lista_rostos.detections:
    for rosto in lista_rostos.detections:
      #desenhar os pontos no rosto.
      desenho.draw_detection(frame, rosto)

  #Ferramenta para visualizar a imagem.
  #Dois argumentos título e qual imagem a ser exibida.

  cv2.imshow("Faces disponíveis", frame)

  #para parar precisa apertar esc.
  #Se a tecla que apertar for igual a 27 interrompa.
  if cv2.waitKey(5) == ord('p'):
    break
#Finalizar o uso da webcam sempre.
webcam.release()
cv2.destroyAllWindows()