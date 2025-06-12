import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

class TensorRTYOLO:
    def __init__(self, engine_path, input_size=640):
        self.input_size = input_size
        self.engine_path = engine_path
        
        self.classes = {
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
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        print(f"TensorRT Engine carregado: {engine_path}")
        print(f"Input shape: {self.inputs[0].host.shape}")
        print(f"Output shape: {self.outputs[0].host.shape}")
    
    def load_engine(self):
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.input_size/w, self.input_size/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        
        image_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        image_padded = image_padded.astype(np.float32) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
        
        return image_padded, scale, dw, dh
    
    def postprocess(self, output, scale, dw, dh, conf_threshold=0.2, nms_threshold=0.4):
        detections = []
        output = output.reshape(1, -1, output.shape[-1])
        
        for detection in output[0]:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold and class_id in self.classes:
                x_center, y_center, width, height = detection[:4]
                x_center = (x_center - dw) / scale
                y_center = (y_center - dh) / scale
                width = width / scale
                height = height / scale
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.classes[class_id],
                    'confidence': float(confidence),
                    'bbox': [x1, y1, x2, y2]
                })
        
        if len(detections) > 1:
            detections = self.apply_nms(detections, nms_threshold)
        
        return detections
    
    def apply_nms(self, detections, nms_threshold):
        if not detections:
            return detections
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        boxes_nms = boxes.copy()
        boxes_nms[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_nms[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), scores.tolist(), 0.5, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return detections
    
    def infer(self, image):
        input_data, scale, dw, dh = self.preprocess(image)
        np.copyto(self.inputs[0].host, input_data.ravel())
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        output = self.outputs[0].host
        detections = self.postprocess(output, scale, dw, dh)
        
        annotated_frame = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return detections, annotated_frame

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def main():
    detector = TensorRTYOLO("best.engine", 416)
    #cap = cv2.VideoCapture("http://100.93.94.123:8080/stream.mjpg")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        exit()
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        inference_start = time.time()
        detections, annotated_frame = detector.infer(frame)
        inference_time = time.time() - inference_start
        
        frame_count += 1
        fps = frame_count / (time.time() - start_time + 0.001)
        
        info_text = f"Objects: {len(detections)} | FPS: {fps:.1f} | Tempo: {inference_time:.3f}s"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
    print(f"Total de frames processados: {frame_count}")

if __name__ == "__main__":
    main()
