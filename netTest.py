import cv2
import time
from ultralytics import YOLO

class YOLORealTimeInference:
    def __init__(self, model_path='best.pt', confidence=0.2):
        print(f"Load Model: {model_path}")
        self.model = YOLO(model_path)
        print(self.model.names)
        self.confidence = confidence
        
        # Classes relevantes para detecção rodoviária
        self.target_classes ={
            0: 'STOP',
            1: 'PASSAGEM', 
            2: 'VEL_50',
            3: 'VEL_80',
            4: 'SEMAFORO_VERMELHO',
            5: 'SEMAFORO_VERDE',
            6: 'SEMAFORO_LARANJA',
            7: 'PASSADEIRA',
            8: 'DANGER',
            9: 'CURVA',
        }

        
        # Cores para diferentes classes (BGR format)
        self.colors = {
            0: (255, 0, 0),    
            1: (0, 255, 0),    
            2: (0, 0, 255),    
            3: (255, 255, 0),  
            4: (255, 0, 255),   
            5: (0, 255, 255),  
            6: (128, 0, 128),  
            7: (0, 128, 255),  
            8: (128, 128, 0),  
            9: (255, 165, 0)

        }
    
    def detect_frame(self, frame):
        start_time = time.time()
        results = self.model(frame, conf=self.confidence, verbose=False)
        inference_time = time.time() - start_time
        
        detections = []
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls in self.target_classes:
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_name = self.target_classes[cls]
                        
                        detection = {
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
                        
                        color = self.colors.get(cls, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (220, 220, 220), 1)  
        
        info_text = f"Objects: {len(detections)} | Tempo: {inference_time:.3f}s"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return detections, annotated_frame

cap = cv2.VideoCapture("http://100.93.94.123:8080/stream.mjpg")
detector = YOLORealTimeInference()
  
 
if not cap.isOpened():
    print("Erro ao abrir vídeo.")
    exit()

 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 

    detections, annotated_frame = detector.detect_frame(frame)
        
    # Redimensionar se muito grande
    h, w = annotated_frame.shape[:2]
    if w > 1200 or h > 800:
        scale = min(1200/w, 800/h)
        new_w, new_h = int(w*scale), int(h*scale)
        annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))
    
    cv2.imshow('YOLO Detection', annotated_frame)
 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
 
cv2.destroyAllWindows()

 

