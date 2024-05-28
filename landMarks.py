import cv2
import mediapipe as mp

# Inicializar o modelo de solução holística e utilitários de desenho
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Abrir a câmera
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Verificar se a imagem está vazia
    if frame is None:
        print("Não foi possível capturar a imagem. Verifique se a câmera está conectada e funcionando corretamente.")
        continue

    # Converter a imagem para RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fazer a previsão
    results = holistic.process(rgb_image)

    # Desenhar os pontos de referência faciais
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        # Obter os pontos de referência dos olhos
        left_eye = results.face_landmarks.landmark[33]
        right_eye = results.face_landmarks.landmark[263]

        # Verificar se os olhos estão no mesmo nível vertical
        if abs(left_eye.y - right_eye.y) < 0.02:
            print("O rosto está olhando para a câmera.")
            break  # Sair do loop e terminar o script

    # Exibir a imagem
    cv2.imshow('Webcam Feed', frame)

    # Se a tecla 'q' for pressionada, sair do loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Fechar a janela da imagem
cap.release()
cv2.destroyAllWindows()
