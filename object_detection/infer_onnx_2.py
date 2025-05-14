import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Configurações
model_path = "onnx_engine/roboflow_003v5_320.onnx"  # Caminho do teu modelo treinado
video_path = "caminho/para/teu/video.mp4"  # Caminho do vídeo (deixa vazio para usar webcam)
output_path = "resultados_video.mp4"  # Caminho para salvar o vídeo processado
class_names = ["Stop", "Zebra", "crosswalk"]  # Classes do teu dataset
conf_threshold = 0.5  # Threshold de confiança
iou_threshold = 0.45  # Threshold de IoU para NMS
img_size = 320  # Tamanho de entrada do modelo (ajusta se necessário)

# Função para pós-processamento (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# Função para processar frame
def process_frame(frame, session):
    # Dimensões originais do frame
    h, w = frame.shape[:2]

    # Pré-processamento
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC para CHW e normalizar
    img = np.expand_dims(img, axis=0)  # Adicionar batch dimension

    # Inferência
    inputs = {session.get_inputs()[0].name: img}
    pred = session.run(None, inputs)[0]  # [batch, num_boxes, 4 + 1 + num_classes]

    # Pós-processamento
    boxes, scores, classes = [], [], []
    for box in pred[0]:
        x1, y1, x2, y2, conf = box[:5]
        cls_scores = box[5:]
        cls = np.argmax(cls_scores)
        if conf > conf_threshold:
            # Normalizar coordenadas para [0, 1] e depois escalar para o tamanho original
            x1, y1, x2, y2 = x1 / img_size, y1 / img_size, x2 / img_size, y2 / img_size
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            classes.append(cls)

    # Aplicar NMS
    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = non_max_suppression(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = np.array(classes)[keep]

        # Desenhar caixas
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Corrigir para não ultrapassar os limites do frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            label = f"{class_names[int(classes[i])]} {scores[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Carregar modelo ONNX
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

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
    frame = process_frame(frame, session)

    # Exibir frame
    cv2.imshow("YOLOv5 ONNX Detections", frame)

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