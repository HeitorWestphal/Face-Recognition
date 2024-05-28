import cv2
import mediapipe as mp

# Inicializar o modelo de solução holística e utilitários de desenho
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Abrir a câmera
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Converter a imagem para RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fazer a previsão
    results = holistic.process(rgb_image)

    # Se a íris for detectada
    if results.face_landmarks:
        # Obter os pontos de referência da íris esquerda e direita
        left_iris = results.face_landmarks.landmark[33:37]  # Índices dos pontos de referência da íris esquerda
        right_iris = results.face_landmarks.landmark[263:267]  # Índices dos pontos de referência da íris direita

        # Desenhar os pontos de referência da íris
        for landmark in left_iris + right_iris:
            cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 1, (0, 255, 0), 1)

    # Exibir a imagem
    cv2.imshow('Webcam Feed', frame)

    # Verificar se a tecla 'q' foi pressionada - se sim, quebrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a janela da imagem
cap.release()
cv2.destroyAllWindows()
