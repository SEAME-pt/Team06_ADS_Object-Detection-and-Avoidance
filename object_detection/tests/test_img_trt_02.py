import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

MODEL_PATH = "../onnx_engine/roboflow002v5_320.engine"
INPUT_SIZE = (320, 320)
CONF_THRES = 0.05
IOU_THRES = 0.5
CLASSES = ['NoEntry', 'Stop-sign', 'Pedestrian Crossing']

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_shape[0] - new_h, new_shape[1] - new_w
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def preprocess_image(image, input_size):
    img = letterbox(image, input_size)
    cv2.imwrite("letterboxed_image.jpg", img)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

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
    print("Saída bruta shape:", output.shape)
    print("Primeiros valores da saída:", output[:10])
    output = output.reshape(1, -1, 5+len(CLASSES))
    print("Saída reshape shape:", output.shape)
    return output

def scale_boxes(boxes, input_shape, original_shape):
    gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
    new_h, new_w = int(original_shape[0] * gain), int(original_shape[1] * gain)
    pad_h = (input_shape[0] - new_h) / 2
    pad_w = (input_shape[1] - new_w) / 2
    print(f"Gain: {gain}, New H: {new_h}, New W: {new_w}, Pad H: {pad_h}, Pad W: {pad_w}")

    scaled_boxes = boxes.copy()
    if len(boxes) > 0:
        scaled_boxes[:, 0] = (boxes[:, 0] - pad_w) / gain  # x (centro)
        scaled_boxes[:, 1] = (boxes[:, 1] - pad_h) / gain  # y (centro)
        scaled_boxes[:, 2] = boxes[:, 2] / gain  # largura
        scaled_boxes[:, 3] = boxes[:, 3] / gain  # altura
    return scaled_boxes

def non_max_suppression(boxes, scores, conf_thres, iou_thres):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    return indices.flatten() if len(indices) > 0 else np.array([])

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
        print(f"Detecção: [x1={x1}, y1={y1}, x2={x2}, y2={y2}], Classe: {class_name}, Score: {score:.2f}")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    engine = load_engine(MODEL_PATH)
    image = cv2.imread("../data_test/img03.jpg")
    if image is None:
        print("Erro ao carregar imagem")
        return
    original_shape = image.shape[:2]
    print("Tamanho original da imagem:", original_shape)
    outputs = infer(engine, image)
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
    print("Caixas antes scale_boxes:", boxes)
    boxes = np.array(boxes) if boxes else np.empty((0, 4))
    scores = np.array(scores) if scores else np.empty((0,))
    class_ids = np.array(class_ids) if class_ids else np.empty((0,))
    boxes = scale_boxes(boxes, INPUT_SIZE, original_shape)
    print("Caixas após scale_boxes:", boxes)
    indices = non_max_suppression(boxes, scores, CONF_THRES, IOU_THRES)
    print("Caixas após NMS:", boxes[indices] if len(indices) > 0 else [])
    if len(indices) > 0:
        image = draw_boxes(image, boxes[indices], scores[indices], class_ids[indices])
    cv2.imwrite("output_image.jpg", image)
    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()