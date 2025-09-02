
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os
from pathlib import Path

 
 

class TensorRTYOLO:
    def __init__(self, engine_path, input_size=352):
        self.input_size = input_size
        self.engine_path = engine_path
        self.classes = {
            0: 'person',
            1: 'car',
            2: 'stop sign',
            3: 'lane',
            4: 'passadeira',
            5: 'sinal verde',
            6: 'sinal amarelo',
            7: 'sinal vermelho',
            8: 'sinal perigo',
            9: 'speed 50',
            10: 'speed 80',
            11: 'jetracer',
            12: 'drivable',
            13: 'prioridade',
            14: 'gate closed'
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
            9: (255, 0, 0),    
            10: (0, 255, 0),    
            11: (0, 0, 255),    
            12: (255, 255, 0),  
            13: (255, 0, 255),   
            14: (0, 255, 255),  
            15: (128, 0, 128),  
            16: (0, 128, 255),  
            17: (128, 128, 0)
        }
        
        # Inicializar TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        # Preparar buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # CORREÇÃO: Verificar shapes reais
        print(f"TensorRT Engine carregado: {engine_path}")
        print(f"Input shape: {self.inputs[0].host.shape}")
        print(f"Output shape: {self.outputs[0].host.shape}")
        
        # CORREÇÃO: Calcular número de classes e anchors baseado no output
        output_size = self.outputs[0].host.shape[0]
        self.num_classes = len(self.classes)
        self.num_anchors = output_size // (self.input_size * self.input_size // 32 // 32) // (5 + self.num_classes)
        print(f"Detectadas {self.num_classes} classes, {self.num_anchors} anchors")

    def postprocess(self, output, scale, dw, dh, conf_threshold=0.25, nms_threshold=0.4):
        """Postprocessing corrigido para YOLOv8"""
        try:
 
            # Output shape esperado: (1, num_classes + 4, num_detections)
            if len(output.shape) == 1:
                # Reshape para formato YOLOv8: (batch, 4+classes, detections)
                num_detections = output.shape[0] // (4 + self.num_classes)
                output = output.reshape(1, 4 + self.num_classes, num_detections)
            
            # Transpor para (batch, detections, 4+classes)
            output = np.transpose(output, (0, 2, 1))[0]  # Remove batch dimension
            
            print(f"Output shape após reshape: {output.shape}")
            print(f"Primeiras 5 detecções (scores): {output[:5, 4:].max(axis=1)}")
            
            # CORREÇÃO: Separar coordenadas e scores
            boxes = output[:, :4]  # x_center, y_center, width, height
            class_scores = output[:, 4:]  # scores para cada classe
            
            # Encontrar classe com maior score para cada detecção
            class_ids = np.argmax(class_scores, axis=1)
            max_scores = np.max(class_scores, axis=1)
            
            # CORREÇÃO: Filtrar por threshold mais baixo
            mask = max_scores > conf_threshold
            
            if not np.any(mask):
                print(f"Nenhuma detecção acima do threshold {conf_threshold}")
                print(f"Score máximo encontrado: {max_scores.max():.4f}")
                return []
            
            # Aplicar filtro
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            scores = max_scores[mask]
            
            print(f"Detecções filtradas: {len(boxes)}")
            
            # CORREÇÃO: Converter coordenadas do centro para x1,y1,x2,y2
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # Ajustar para coordenadas da imagem original
            x_center = (x_center - dw) / scale
            y_center = (y_center - dh) / scale
            width = width / scale
            height = height / scale
            
            # Converter para formato x1, y1, x2, y2
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Criar array de boxes para NMS
            boxes_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h para cv2.dnn.NMSBoxes
            
            # Aplicar NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(), 
                scores.tolist(), 
                conf_threshold, 
                nms_threshold
            )
            
            detections = []
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices[0]) > 0 else []
                
                for i in indices.flatten():
                    detection = {
                        'class_id': int(class_ids[i]),
                        'class_name': self.classes.get(int(class_ids[i]), 'unknown'),
                        'confidence': float(scores[i]),
                        'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
                    }
                    detections.append(detection)
                    print(f"Detecção: {detection['class_name']} - {detection['confidence']:.3f}")
            
            return detections
            
        except Exception as e:
            print(f"Erro no postprocessing: {e}")
            import traceback
            traceback.print_exc()
            return []

    def infer(self, image):
        """Inferência com debug adicional"""
        try:
            input_data, scale, dw, dh = self.preprocess(image)
            
            # Debug: verificar dados de entrada
            print(f"Input data range: [{input_data.min():.3f}, {input_data.max():.3f}]")
            
            # Copiar para buffer de input
            np.copyto(self.inputs[0].host, input_data.ravel())
            
            # Transferir para GPU
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            
            # Executar inferência
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                print("Falha na execução da inferência")
                return []
            
            # Transferir resultado de volta
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            
            # Sincronizar
            self.stream.synchronize()
            
            output = self.outputs[0].host
            
            # Debug: verificar output
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output shape: {output.shape}")
            
            # Postprocessing com threshold mais baixo
            detections = self.postprocess(output, scale, dw, dh, conf_threshold=0.1, nms_threshold=0.4)
            
            return detections
            
        except Exception as e:
            print(f"Erro durante inferência: {e}")
            import traceback
            traceback.print_exc()
            return []

    def load_engine(self):
        """Carregar TensorRT engine"""
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        """Alocar buffers GPU/CPU"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Alocar host e device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        """Preprocessar imagem para inferência"""
        h, w = image.shape[:2]
        
        # Resize mantendo aspect ratio
        scale = min(self.input_size/w, self.input_size/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        
        # Padding
        image_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        # Normalizar e converter para formato TensorRT
        image_padded = image_padded.astype(np.float32) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
        
        return image_padded, scale, dw, dh
    
 
    
    def apply_nms(self, detections, nms_threshold):
        """Aplicar Non-Maximum Suppression"""
        if not detections:
            return detections
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Converter para formato (x1, y1, w, h)
        boxes_nms = boxes.copy()
        boxes_nms[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_nms[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(), 0.5, nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return detections
    
 

class HostDeviceMem:
 
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def setup_csi_camera(width=640, height=480, fps=30, flip_method=0):
    gst_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR !  appsink drop=1 max-buffers=1"
    )

    #std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
    #                       "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    #                       "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)


def main():
  
    
    
        print("Inicializando TensorRT...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"GPU disponível: {torch.cuda.is_available()}")
        print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Memória GPU livre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

        detector = TensorRTYOLO("model.engine", 352)
        
        print("Configurando camera CSI...")
        #cap = setup_csi_camera(640, 480, 30, 0)
        #cap = cv2.VideoCapture("http://100.93.94.123:8080/stream.mjpg")
        cap = cv2.VideoCapture(0)
        
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a camera CSI")
            print("Verifique se a camera está conectada corretamente")
            return
        
        print("Camera CSI inicializada com sucesso!")
        print("Pressione 'q' para sair")
        
 
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame")
                    break
                
                # Inferência
                inference_start = time.time()
                detections = detector.infer(frame)
                inference_time = time.time() - inference_start
                
   
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    class_id = det['class_id']
                    confidence = det['confidence']
                    class_name = det['class_name']
                    
                    # Bounding box
                    color = detector.colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
 
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = 30 / elapsed
                    start_time = time.time()
                    
                    print(f"FPS: {current_fps:.1f} | Inference: {inference_time*1000:.1f}ms | Detections: {len(detections)}")
                
 
                info_text = f"FPS: {fps_counter/(time.time()-start_time+0.001):.1f} | Detections: {len(detections)}"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
 
                cv2.imshow('TensorRT YOLO - CSI Camera', frame)
                
                frame_count += 1
                
              
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrompido")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nTotal de frames processados: {frame_count}")

main()
