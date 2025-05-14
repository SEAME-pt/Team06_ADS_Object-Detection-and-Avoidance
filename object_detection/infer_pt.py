import torch
import cv2
from pathlib import Path
import sys
import numpy as np

# Adicionar o diretório do YOLOv5 ao sys.path
yolov5_path = "../../yolov5"  # Caminho para o diretório clonado do YOLOv5
sys.path.append(str(Path(yolov5_path).resolve()))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

# Configurações
model_path = "pt/od_v3_416.pt"  # Caminho do teu modelo treinado
video_path = "caminho/para/teu/video.mp4"  # Caminho do vídeo (deixa vazio para usar webcam)
output_path = "resultados_video.mp4"  # Caminho para salvar o vídeo processado
class_names = ['NoEntry', 'stop-sign']  # Classes do teu dataset
conf_threshold = 0.5  # Threshold de confiança
iou_threshold = 0.45  # Threshold de IoU para NMS

# Carregar o modelo treinado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights=model_path, device=device, dnn=False)
model.eval()

# Função para processar frame
def process_frame(frame, model):
    # Converter para RGB e preparar para o modelo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame_rgb).to(device)
    img = img.permute(2, 0, 1).float() / 255.0  # HWC para CHW e normalizar
    img = img.unsqueeze(0)  # Adicionar batch dimension

    # Fazer inferência
    with torch.no_grad():
        pred = model(img)[0]

    # Aplicar NMS
    pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=iou_threshold)[0]

    # Processar deteções
    if pred is not None and len(pred):
        for det in pred:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Configurar entrada de vídeo
#if video_path:
#    cap = cv2.VideoCapture(video_path)
#else:
#    cap = cv2.VideoCapture(0)  # Webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera. Verifica a conexão ou o índice (0, 1, etc.).")
    exit()

# Verificar se a captura foi aberta
#if not cap.isOpened():
#    print("Erro: Não foi possível abrir o vídeo ou webcam.")
#    print("1. Verifique se a webcam está conectada ou o caminho do vídeo está correto.")
#    print("2. Para webcam, tente outro índice (ex.: 1 ou 2).")
#    exit()

# Configurar saída de vídeo (se for vídeo)
if video_path:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
else:
    out = None

# Processar frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processar frame
    frame = process_frame(frame, model)

    # Exibir frame
    cv2.imshow("YOLOv5 Detections", frame)

    # Salvar frame no vídeo de saída (se aplicável)
    if out:
        out.write(frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

print(f"Processamento concluído. Resultados salvos em: {output_path if video_path else 'Webcam'}")