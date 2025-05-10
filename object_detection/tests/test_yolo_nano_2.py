import tensorrt as trt
print("tensorrt ok")
import time
print("time ok")
import pycuda.driver as cuda
print("pycuda ok")
import pycuda.autoinit
print("cuda autoinit ok")
import cv2
print("cv2 ok")
import numpy as np
print("numpy ok")

# Configurações
MODEL_PATH = "best_416.engine"  # Ou "yolov5n.engine" para mais FPS
INPUT_SIZE = (416, 416)  # Reduzido para maior FPS
CONF_THRES = 0.4  # Ajustado para mais detecções
IOU_THRES = 0.5  # Ajustado para menos falsos positivos
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Função de NMS otimizada
def non_max_suppression(boxes, scores, conf_thres, iou_thres):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    return indices

# Carregar motor TensorRT
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Pré-processamento otimizado
def preprocess_image(image, input_size):
    img = cv2.resize(image, input_size, interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

# Inferência com TensorRT
def infer(engine, image):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    input_image = preprocess_image(image, INPUT_SIZE)
    np.copyto(inputs[0]["host"], input_image.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    output = outputs[0]["host"].reshape(-1, 85)  # Ajuste se o formato for diferente
    return output


# Desenhar caixas com nomes de classes
def draw_boxes(image, boxes, scores, class_ids):
    for i, idx in enumerate(class_ids):
        box = boxes[i]
        score = scores[i]
        class_id = int(class_ids[i])
        class_name = CLASSES[class_id]
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


# Aquecimento CUDA
def warm_up(engine, input_size):
    context = engine.create_execution_context()
    dummy_input = np.random.rand(1, 3, input_size[0], input_size[1]).astype(np.float32)
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    np.copyto(inputs[0]["host"], dummy_input.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

def main():
    # Carregar motor TensorRT
    engine = load_engine(MODEL_PATH)
    
    # Aquecimento CUDA
    warm_up(engine, INPUT_SIZE)
    
    # Configurar captura de vídeo

    cap = cv2.VideoCapture(
        "nvarguscamerasrc sensor-mode=5 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=416, height=416, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return

    skip_frame = 4  # Processar 1 a cada 2 frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        frame_count += 1

        if frame_count % skip_frame != 0:
            # Só exibe a imagem sem processar
            cv2.imshow("YOLOv5 TensorRT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Realizar inferência
        start_time = time.time()
        outputs = infer(engine, frame)

        # Processar saída
        boxes, scores, class_ids = [], [], []
        for pred in outputs:
            conf = pred[4]
            if conf > CONF_THRES:
                x, y, w, h = pred[0:4]
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                if class_score * conf > CONF_THRES:
                    boxes.append([x, y, w, h])
                    scores.append(conf * class_score)
                    class_ids.append(class_id)

        # Aplicar NMS
        indices = non_max_suppression(np.array(boxes), np.array(scores), CONF_THRES, IOU_THRES)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = np.array(boxes)[indices]
            scores = np.array(scores)[indices]
            class_ids = np.array(class_ids)[indices]

        # Desenhar caixas
        frame = draw_boxes(frame, boxes, scores, class_ids)

        # Mostrar FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibir frame
        cv2.imshow("YOLOv5 TensorRT", frame)

        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()