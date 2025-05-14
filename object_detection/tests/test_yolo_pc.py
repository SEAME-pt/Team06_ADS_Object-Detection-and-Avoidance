import torch
import cv2
import numpy as np
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.models.common import DetectMultiBackend

# Configurações
WEIGHTS = "runs/train/exp11/weights/best.pt"  # Caminho para o teu ficheiro .pt
IMG_SIZE = 320  # Tamanho de entrada do modelo (mesmo usado no treino)
CONF_THRES = 0.25  # Limiar de confiança
IOU_THRES = 0.45  # Limiar de IoU para NMS
CLASSES = ['NoEntry', 'Stop-sign', 'Pedestrian Crossing']  # Classes do teu dataset

# Inicializa a câmera (índice 0 para webcam padrão)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera. Verifica a conexão ou o índice (0, 1, etc.).")
    exit()

# Define a resolução da câmera (ajusta conforme necessário)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Carrega o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(WEIGHTS, device=device)
model.eval()

# Função de pré-processamento (letterbox)
def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_shape[0] - new_h, new_shape[1] - new_w
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, (left, top))

# Loop de captura e inferência
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # Pré-processa o frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    img, (gain, (pad_x, pad_y)) = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Adiciona dimensão do batch

    # Realiza a inferência
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]

    # Processa as deteções
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], (original_h, original_w)).round()
        for *xyxy, conf, cls in pred:
            label = f"{CLASSES[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)
    else:
        print("Nenhuma deteção encontrada neste frame.")

    # Exibe o frame com deteções
    cv2.imshow("Deteções em Tempo Real", frame)

    # Sai com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()

print("Teste concluído.")