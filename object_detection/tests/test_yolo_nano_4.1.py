import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

# Configurações
MODEL_PATH = "../onnx_engine/stop_noEntry.engine"
INPUT_SIZE = (416, 416)
CONF_THRES = 0.2  # Reduzido para capturar mais detecções
IOU_THRES = 0.5
CLASSES = ['NoEntry', 'stop-sign']

# Função de escalonamento de caixas
def scale_boxes(boxes, input_shape, original_shape):
    """Escala as caixas de input_shape (ex.: 416x416) para original_shape (ex.: 720x1280)."""
    gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
    pad_x = (input_shape[1] - original_shape[1] * gain) / 2
    pad_y = (input_shape[0] - original_shape[0] * gain) / 2
    
    scaled_boxes = boxes.copy()
    if len(boxes) > 0:
        scaled_boxes[:, 0] = (boxes[:, 0] - pad_x) / gain  # x
        scaled_boxes[:, 1] = (boxes[:, 1] - pad_y) / gain  # y
        scaled_boxes[:, 2] = boxes[:, 2] / gain  # w
        scaled_boxes[:, 3] = boxes[:, 3] / gain  # h
    return scaled_boxes

# Função de NMS
def non_max_suppression(boxes, scores, conf_thres, iou_thres):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    return indices.flatten() if len(indices) > 0 else np.array([])

# Carregar motor TensorRT
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Pré-processamento
def preprocess_image(image, input_size):
    img = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# Inferência
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
    
    output = outputs[0]["host"]
    print("Shape da saída bruta:", output.shape)
    output = output.reshape(1, -1, 5+len(CLASSES))
    print("Shape após reshape:", output.shape)
    return output

# Desenhar caixas
def draw_boxes(image, boxes, scores, classes):
    for i, idx in enumerate(classes):
        box = boxes[i]
        score = scores[i]
        class_id = int(classes[i])
        class_name = CLASSES[class_id]
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        print(f"Caixa: [{x1}, {y1}, {x2}, {y2}], Classe: {class_name}, Score: {score:.2f}")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    engine = load_engine(MODEL_PATH)
    warm_up(engine, INPUT_SIZE)
    
    # Teste com imagem estática (descomente para testar)
    # image = cv2.imread("test_image.jpg")  # Substitua por uma imagem com sinais
    # original_shape = image.shape[:2]
    # outputs = infer(engine, image)
    # boxes, scores, class_ids = [], [], []
    # for pred in outputs[0]:
    #     conf = pred[4]
    #     if conf > CONF_THRES:
    #         x, y, w, h = pred[0:4]
    #         class_scores = pred[5:5+len(CLASSES)]
    #         class_id = np.argmax(class_scores)
    #         class_score = class_scores[class_id]
    #         if class_score * conf > CONF_THRES:
    #             boxes.append([x, y, w, h])
    #             scores.append(conf * class_score)
    #             class_ids.append(class_id)
    # print("Caixas antes do NMS:", len(boxes))
    # boxes = np.array(boxes) if boxes else np.empty((0, 4))
    # scores = np.array(scores) if scores else np.empty((0,))
    # class_ids = np.array(class_ids) if class_ids else np.empty((0,))
    # boxes = scale_boxes(boxes, INPUT_SIZE, original_shape)
    # indices = non_max_suppression(boxes, scores, CONF_THRES, IOU_THRES)
    # if len(indices) > 0:
    #     image = draw_boxes(image, boxes[indices], scores[indices], class_ids[indices])
    # cv2.imshow("Resultado", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return

    # Captura de vídeo
    cap = cv2.VideoCapture(
        "nvarguscamerasrc sensor-mode=5 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=416, height=416, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )
    
    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return
    
    skip_frame = 2
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
        
        original_shape = frame.shape[:2]
        start_time = time.time()
        outputs = infer(engine, frame)
        
        boxes, scores, class_ids = [], [], []
        for pred in outputs[0]:
            conf = pred[4]
            if conf > CONF_THRES:
                x, y, w, h = pred[0:4]
                class_scores = pred[5:5+len(CLASSES)]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                if class_score * conf > CONF_THRES:
                    boxes.append([x, y, w, h])
                    scores.append(conf * class_score)
                    class_ids.append(class_id)
        
        print("Caixas antes do NMS:", len(boxes))
        boxes = np.array(boxes) if boxes else np.empty((0, 4))
        scores = np.array(scores) if scores else np.empty((0,))
        class_ids = np.array(class_ids) if class_ids else np.empty((0,))
        boxes = scale_boxes(boxes, INPUT_SIZE, original_shape)
        indices = non_max_suppression(boxes, scores, CONF_THRES, IOU_THRES)
        
        if len(indices) > 0:
            frame = draw_boxes(frame, boxes[indices], scores[indices], class_ids[indices])
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("YOLOv5 TensorRT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()