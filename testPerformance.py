import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os


import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os

class TensorRTYOLO:
    def __init__(self, engine_path, input_size=416):
        """
        Inicializar TensorRT YOLO
        
        Args:
            engine_path: Caminho para o arquivo .engine
            input_size: Tamanho do input (320, 416, 640)
        """
        self.input_size = input_size
        self.engine_path = engine_path
    

        self.classes ={
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
        
        # Inicializar TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        # Preparar buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        print(f"TensorRT Engine carregado: {engine_path}")
        print(f"Input shape: {self.inputs[0].host.shape}")
        print(f"Output shape: {self.outputs[0].host.shape}")
    
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
    
    def postprocess(self, output, scale, dw, dh, conf_threshold=0.5, nms_threshold=0.4):
        """Postprocessar saída da rede"""
        detections = []
        
        # Reshape output (assumindo YOLOv8 format)
        output = output.reshape(1, -1, output.shape[-1])
        
        for detection in output[0]:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold and class_id in self.classes:
                # Bounding box (center format)
                x_center, y_center, width, height = detection[:4]
                
                # Converter para coordenadas da imagem original
                x_center = (x_center - dw) / scale
                y_center = (y_center - dh) / scale
                width = width / scale
                height = height / scale
                
                # Converter para x1, y1, x2, y2
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
            
        
        # Aplicar NMS se necessário
        if len(detections) > 1:
            detections = self.apply_nms(detections, nms_threshold)
        
        return detections
    
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
    
    def infer(self, image):
        """Fazer inferência numa imagem"""
        # Preprocessar
        input_data, scale, dw, dh = self.preprocess(image)
        
        # Copiar para buffer de input
        np.copyto(self.inputs[0].host, input_data.ravel())
        
        # Transferir para GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        
        # Executar inferência
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transferir resultado de volta
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        
        # Sincronizar
        self.stream.synchronize()
        
        # Postprocessar
        output = self.outputs[0].host
        detections = self.postprocess(output, scale, dw, dh)
        
        return detections

class HostDeviceMem:
    """Classe auxiliar para gerenciar memória host/device"""
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

 
    return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

def main():
    detector = TensorRTYOLO("best.engine", 320)
    #cap = setup_csi_camera(640, 480, 30, 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a camera CSI")
        return

    print("Teste de performance iniciado.  Ctrl+C para parar.")
    
    total_frames = 0
    total_inference_time = 0.0
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
            total_inference_time += inference_time
            total_frames += 1

            # Imprime status a cada 30 frames
            if total_frames % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = total_frames / elapsed
                avg_inference = total_inference_time / total_frames * 1000
                print(f"Frames: {total_frames} | FPS: {current_fps:.2f} | Tempo médio de inferência: {avg_inference:.2f}ms | Detecções: {len(detections)}")

    except KeyboardInterrupt:
        print("\nTeste finalizado pelo usuário.")
    finally:
        cap.release()
        elapsed = time.time() - start_time
        avg_fps = total_frames / elapsed if elapsed > 0 else 0
        avg_inference = total_inference_time / total_frames * 1000 if total_frames > 0 else 0
        print(f"\nResultados finais:")
        print(f"- Frames processados: {total_frames}")
        print(f"- Tempo total: {elapsed:.2f}s")
        print(f"- FPS médio: {avg_fps:.2f}")
        print(f"- Tempo médio de inferência: {avg_inference:.2f}ms")
        print(f"- Detecções médias por frame: {len(detections) if total_frames > 0 else 0}")

if __name__ == "__main__":
    main()
