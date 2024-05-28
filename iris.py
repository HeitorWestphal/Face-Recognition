import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from PIL import Image, ImageFilter

# Função para melhorar a qualidade da imagem
def improve_image_quality(image):
    # Converta a imagem do OpenCV para uma imagem PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Aplique alguns filtros para melhorar a qualidade da imagem
    pil_image = pil_image.filter(ImageFilter.SHARPEN)
    pil_image = pil_image.filter(ImageFilter.DETAIL)

    # Converta a imagem PIL de volta para uma imagem OpenCV
    improved_image = np.array(pil_image)
    improved_image = cv2.cvtColor(improved_image, cv2.COLOR_RGB2BGR)

    return improved_image

# Função para salvar a imagem em um banco de dados SQLite
def save_image_to_db(image, db_name, table_name):
    # Conecte-se ao banco de dados
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Crie a tabela se ela não existir
    c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, image BLOB)")

    # Converta a imagem em bytes
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    # Insira a imagem no banco de dados
    c.execute(f"INSERT INTO {table_name} (image) VALUES (?)", (image_bytes,))

    # Confirme as alterações e feche a conexão
    conn.commit()
    conn.close()

# Função para recuperar imagens do banco de dados
def get_images_from_db(db_name, table_name):
    # Conecte-se ao banco de dados
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Recupere todas as imagens do banco de dados
    c.execute(f"SELECT * FROM {table_name}")
    rows = c.fetchall()

    # Feche a conexão com o banco de dados
    conn.close()

    # Converta as imagens de volta para um formato que o OpenCV possa usar
    images = [cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR) for row in rows]

    return images

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

        # Calcular a região de interesse (ROI) para o zoom
        iris_landmarks = left_iris + right_iris
        x_min = int(min(landmark.x for landmark in iris_landmarks) * frame.shape[1])
        x_max = int(max(landmark.x for landmark in iris_landmarks) * frame.shape[1])
        y_min = int(min(landmark.y for landmark in iris_landmarks) * frame.shape[0])
        y_max = int(max(landmark.y for landmark in iris_landmarks) * frame.shape[0])

        # Ajustar a ROI para evitar sair dos limites da imagem
        x_min = max(0, x_min - 10)  # Adicionar margem para diminuir o zoom
        y_min = max(0, y_min - 30)  # Adicionar margem para diminuir o zoom
        x_max = min(frame.shape[1], x_max + 50)  # Adicionar margem para diminuir o zoom
        y_max = min(frame.shape[0], y_max + 50)  # Adicionar margem para diminuir o zoom

        # Obter a ROI e redimensioná-la
        roi = frame[y_min:y_max, x_min:x_max]
        roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]))

        # Melhorar a qualidade da imagem
        improved_roi = improve_image_quality(roi)

        # Salvar a imagem melhorada no banco de dados
        save_image_to_db(improved_roi, 'my_database.db', 'my_table')

        # Exibir a imagem com zoom
        cv2.imshow('Webcam Feed', improved_roi)
    else:
        # Exibir a imagem original se a íris não for detectada
        cv2.imshow('Webcam Feed', frame)

    # Verificar se a tecla 'q' foi pressionada - se sim, quebrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a janela da imagem
cap.release()
cv2.destroyAllWindows()

# Recuperar e exibir as imagens do banco de dados
images = get_images_from_db('my_database.db', 'my_table')
for i, image in enumerate(images):
    cv2.imshow(f'Image {i+1}', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
