import cv2
import numpy as np
import onnxruntime as ort

# CONFIG
model_path = "onnx/od_v3_416.onnx"
video_path = ""  # "" para webcam ou caminho para vídeo
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
def process_frame(frame, session):
    h0, w0 = frame.shape[:2]

    # Pré-processamento
    img = cv2.resize(frame, (img_size, img_size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]  # (1, 3, 320, 320)

    # Inference
    pred = session.run(None, {session.get_inputs()[0].name: img_input})[0]
    pred = pred[0]

    boxes, scores, class_ids = [], [], []

  
    scale_x = w0 / img_size
    scale_y = h0 / img_size

    for det in pred:
        xc, yc, w, h, conf = det[:5]
        if conf < conf_threshold:
            continue
        cls = np.argmax(det[5:])
        
        
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


cap = cv2.VideoCapture(0 if video_path == "" else video_path)
session = ort.InferenceSession(model_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame, session)
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
