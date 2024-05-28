"""import cv2

# Carregar o classificador pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir a câmera
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da câmera
    _, img = cap.read()

    # Verificar se a imagem está vazia
    if img is None:
        print("Não foi possível capturar a imagem. Verifique se a câmera está conectada e funcionando corretamente.")
        continue

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar equalização de histograma
    gray = cv2.equalizeHist(gray)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Desenhar retângulos ao redor dos rostos e adicionar rótulos
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f'Pessoa {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # Exibir a imagem
    cv2.imshow('img', img)

    # Se a tecla 'q' for pressionada, sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a janela da imagem
cap.release()
cv2.destroyAllWindows()         """
