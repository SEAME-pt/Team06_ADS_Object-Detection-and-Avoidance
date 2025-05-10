import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

# Configurações
MODEL_PATH = "yolov5_crosswalk.engine"  # Caminho do motor TensorRT
INPUT_SIZE = (320, 320)  # Tamanho de entrada
CONF_THRES = 0.4  # Limiar de confiança
IOU_THRES = 0.5  # Limiar de IoU para NMS
CLASSES = ['crosswalk', 'person']  # Classes do dataset

# Função de NMS otimizada
def non_max_suppression(boxes, scores, conf_thres, iou_thres):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    return indices.flatten() if len(indices) > 0 else np.array([])

# Carregar motor TensorRT
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Pré-processamento otimizado
def preprocess_image(image, input_size):
    img = cv2.resize(image, input_size, interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Adiciona batch
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
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    
    # Ajustar para o formato de saída do YOLOv5 (depende do modelo)
    output = outputs[0]["host"].reshape(1, -1, 7)  # [batch, num_boxes, (x, y, w, h, conf, cls0, cls1)]
    return output

# Desenhar caixas com nomes de classes
def draw_boxes(image, boxes, scores, classes):
    for i, idx in enumerate(classes):
        box = boxes[i]
        score = scores[i]
        class_id = int(classes[i])
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

    skip_frame = 2  # Processar 1 a cada 2 frames para maior FPS
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        frame_count += 1
        if frame_count % skip_frame != 0:
            cv2.imshow("YOLOv5 TensorRT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Realizar inferência
        start_time = time.time()
        outputs = infer(engine, frame)

        # Processar saída
        boxes, scores, class_ids = [], [], []
        for pred in outputs[0]:
            conf = pred[4]
            if conf > CONF_THRES:
                x, y, w, h = pred[0:4]
                class_scores = pred[5:7]  # Apenas 2 classes (crosswalk, person)
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                if class_score * conf > CONF_THRES:
                    boxes.append([x, y, w, h])
                    scores.append(conf * class_score)
                    class_ids.append(class_id)

        # Aplicar NMS
        boxes = np.array(boxes) if boxes else np.empty((0, 4))
        scores = np.array(scores) if scores else np.empty((0,))
        class_ids = np.array(class_ids) if class_ids else np.empty((0,))
        indices = non_max_suppression(boxes, scores, CONF_THRES, IOU_THRES)

        # Desenhar caixas
        if len(indices) > 0:
            frame = draw_boxes(frame, boxes[indices], scores[indices], class_ids[indices])

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