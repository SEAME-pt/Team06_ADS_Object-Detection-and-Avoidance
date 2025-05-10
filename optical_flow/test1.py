import cv2
import numpy as np
import os

# 🛑 Forçar o uso de GTK em vez de Qt para evitar problemas de threading
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Força a plataforma XCB, o que pode evitar o erro

# 🔹 Parâmetros para o detetor Shi-Tomasi (encontra bons cantos para seguir)
feature_params = dict(
    maxCorners=100,        # Número máximo de pontos a detetar
    qualityLevel=0.3,      # Qualidade mínima (0 a 1)
    minDistance=7,         # Distância mínima entre cantos detetados
    blockSize=7            # Tamanho da área analisada em volta de cada pixel
)

# 🔹 Parâmetros do algoritmo Lucas-Kanade
lk_params = dict(
    winSize=(15, 15),      # Tamanho da janela de busca (maior = mais robusto, mais lento)
    maxLevel=2,            # Níveis de pirâmide para lidar com movimento grande
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Critério de paragem
)

# 🔹 Inicia a captura de vídeo (0 = webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Erro ao abrir a câmara")
    exit()

# 🔹 Lê o primeiro frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 🔹 Define a Região de Interesse (ROI) — retângulo onde vais procurar pontos
x, y, w, h = 200, 200, 200, 200   # ROI em (x, y) com largura w e altura h
roi = old_gray[y:y+h, x:x+w]      # Recorta a imagem para esta zona

# 🔹 Deteta cantos apenas dentro da ROI
p0 = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)

# 🔹 Ajusta os pontos encontrados para as coordenadas da imagem original
if p0 is not None:
    p0 += np.array([[x, y]], dtype=np.float32)

# 🔹 Máscara para desenhar as trajetórias (vai acumulando as linhas)
mask = np.zeros_like(old_frame)

# 🔄 Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔹 Calcula o movimento (optical flow) dos pontos
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 🔹 Verifica se há pontos válidos
    if p1 is not None and st is not None:
        # Seleciona apenas os pontos bem seguidos
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 🔹 Para cada par de pontos (antigo e novo)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Desenha linha da trajetória
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            # Marca o ponto atual com um círculo
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Combina o vídeo com a máscara de trajetórias
        img = cv2.add(frame, mask)
    else:
        img = frame  # Se não houver movimento, só mostra o vídeo

    # 🔹 Desenha o retângulo da ROI original (só visual)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 🔹 Mostra o resultado
    cv2.imshow('Lucas-Kanade Optical Flow', img)

    # Tecla ESC (27) para sair
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # 🔹 Atualiza o frame e os pontos para o próximo ciclo
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 🖼️ Salva as imagens para debug (caso o `imshow` não funcione)
    cv2.imwrite("frame_debug.jpg", img)

# 🔹 Finaliza tudo
cap.release()
cv2.destroyAllWindows()
