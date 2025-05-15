import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# CONFIG
model_path = "./engine/od_v3_416_nano.trt"  # Caminho do modelo TensorRT
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
def process_frame(frame, engine, context, d_input, d_output, output_shape):
    h0, w0 = frame.shape[:2]

    # Pré-processamento
    img = cv2.resize(frame, (img_size, img_size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]  # (1, 3, 416, 416)
    img_input = np.ascontiguousarray(img_input)

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

# Criar contexto TensorRT
context = engine.create_execution_context()

# Alocar memória CUDA
input_shape = (1, 3, img_size, img_size)
output_shape = (1, 25200, 7)  # [num_boxes, x1, y1, x2, y2, conf, cls]
img_input = np.zeros(input_shape, dtype=np.float32)
d_input = cuda.mem_alloc(img_input.nbytes)
# Converter explicitamente para int Python
output_size_bytes = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(output_size_bytes)

# Configurar entrada de vídeo com pipeline GStreamer
cap = cv2.VideoCapture(
    "nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=416, height=416, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
    "appsink sync=false drop=true",
    cv2.CAP_GSTREAMER
)

# Verificar se a captura foi aberta
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# Inicializar máscara
ret, mask = cap.read()
if not ret:
    print("Erro: Não foi possível ler o primeiro frame.")
    cap.release()
    exit()

# Processar frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Falha ao capturar frame.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            mask = process_frame(frame, engine, context, d_input, d_output, output_shape)
            print("CAP")

        # Exibir frames
        cv2.imshow("Detections", frame)
        cv2.imshow("Object", mask)

        # Sair com 'q'
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("Processamento interrompido pelo usuário.")

finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    d_input.free()
    d_output.free()
    del context
    del engine
    print("Recursos liberados.")