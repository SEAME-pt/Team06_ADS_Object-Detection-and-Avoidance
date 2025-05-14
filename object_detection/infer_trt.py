import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ctypes import c_size_t

# CONFIG
model_path = "./engine/od_v3_416.trt"  # Caminho do modelo TensorRT
video_path = ""  # "" para webcam ou caminho para vídeo
output_path = "resultados_video_trt.mp4"  # Caminho para salvar o vídeo processado
class_names = ['NoEntry', 'stop-sign']
conf_threshold = 0.5
iou_threshold = 0.5
img_size = 416

# NMS
def non_max_suppression(boxes, scores, iou_threshold):
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
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
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return keep

# INFERÊNCIA
def process_frame(frame, engine):
    h0, w0 = frame.shape[:2]

    # Pré-processamento
    img = cv2.resize(frame, (img_size, img_size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]  # (1, 3, 416, 416)
    # Garantir que o array seja contíguo
    img_input = np.ascontiguousarray(img_input)

    # Inferência TensorRT
    with engine.create_execution_context() as context:
        # Alocar memória para entrada e saída
        input_shape = (1, 3, img_size, img_size)
        output_shape = (1, 25200, 7)  # [num_boxes, x1, y1, x2, y2, conf, cls]
        d_input = cuda.mem_alloc(c_size_t(img_input.nbytes).value)
        d_output = cuda.mem_alloc(c_size_t(np.prod(output_shape) * 4).value)

        # Transferir dados para a GPU
        cuda.memcpy_htod(d_input, img_input)
        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings)

        # Transferir resultados da GPU para CPU
        output = np.zeros(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)

    # Pós-processamento
    boxes, scores, class_ids = [], [], []

    scale_x = w0 / img_size
    scale_y = h0 / img_size

    for det in output[0]:
        xc, yc, w, h, conf, *cls_scores = det
        if conf < conf_threshold:
            continue
        cls = np.argmax(cls_scores)
        if cls >= len(class_names):
            continue

        # Escalar para as dimensões originais do frame
        x1 = (xc - w / 2) * scale_x
        y1 = (yc - h / 2) * scale_y
        x2 = (xc + w / 2) * scale_x
        y2 = (yc + h / 2) * scale_y

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cls)

    if not boxes:
        return frame

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = non_max_suppression(boxes, scores, iou_threshold)
    for i in keep:
        x1, y1, x2, y2 = map(int, boxes[i])
        label = f"{class_names[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Carregar modelo TensorRT
with open(model_path, "rb") as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())

# Configurar entrada de vídeo
cap = cv2.VideoCapture(0 if video_path == "" else video_path)

# Verificar se a captura foi aberta
if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo ou webcam.")
    print("1. Verifique se a webcam está conectada ou o caminho do vídeo está correto.")
    print("2. Para webcam, tente outro índice (ex.: 1 ou 2).")
    exit()

# Configurar saída de vídeo (se for vídeo)
if video_path:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
else:
    out = None

# Processar frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame, engine)

    # Exibir frame
    cv2.imshow("Detections", frame)

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