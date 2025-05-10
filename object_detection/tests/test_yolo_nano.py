import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

# Configurações
MODEL_PATH = "best.engine"  # Caminho para o motor TensorRT
INPUT_SIZE = (640, 640)  # Tamanho da entrada do modelo (mesmo usado no treinamento)
CONF_THRES = 0.5  # Limiar de confiança
IOU_THRES = 0.4  # Limiar de IoU para NMS
CLASSES = 80  # Número de classes (COCO tem 80)

# Função para pós-processamento (NMS)
def non_max_suppression(boxes, scores, conf_thres, iou_thres):
    # Implementação simplificada de NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_thres, iou_thres
    )
    return indices

# Carregar motor TensorRT
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Pré-processar imagem
def preprocess_image(image, input_size):
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0  # Normalizar
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Adicionar batch
    return img

# Inferência com TensorRT
def infer(engine, image):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

    # Alocar buffers
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

    # Pré-processar imagem
    input_image = preprocess_image(image, INPUT_SIZE)
    np.copyto(inputs[0]["host"], input_image.ravel())

    # Transferir dados para GPU
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

    # Executar inferência
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # Transferir resultados de volta
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()

    # Processar saída (depende do formato do YOLOv5)
    output = outputs[0]["host"].reshape(-1, 85)  # Exemplo: [N, 85] (x, y, w, h, conf, 80 classes)
    return output

# Desenhar caixas de detecção
def draw_boxes(image, boxes, scores, classes):
    for i, idx in enumerate(classes):
        box = boxes[i]
        score = scores[i]
        class_id = int(classes[i])
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    # Carregar motor TensorRT
    engine = load_engine(MODEL_PATH)
    
    # Configurar captura de vídeo com GStreamer (para câmera CSI do Jetson Nano)
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        # Realizar inferência
        start_time = time.time()
        outputs = infer(engine, frame)
        
        # Processar saída (exemplo simplificado para YOLOv5)
        boxes, scores, class_ids = [], [], []
        for pred in outputs:
            conf = pred[4]  # Confiança
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