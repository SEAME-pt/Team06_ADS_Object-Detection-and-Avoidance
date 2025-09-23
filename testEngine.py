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

class TensorRTWebcamInference:
    def __init__(self, engine_path='best.engine', confidence=0.5):
        print(f"Carregando engine TensorRT: {engine_path}")
        
        self.confidence = confidence
        self.engine_path = engine_path
        
        # Inicializar TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Carregar engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Configurar bindings
        self.setup_bindings()
        
        # Classes e cores (mantidas do código original)
        self.target_classes = {
            0: 'STOP', 1: 'PASSAGEM', 2: 'VEL_50', 3: 'VEL_80',
            4: 'SEMAFORO_VERMELHO', 5: 'SEMAFORO_VERDE', 6: 'SEMAFORO_LARANJA',
            7: 'PASSADEIRA', 8: 'DANGER', 9: 'CURVA'
        }
        
        self.colors = {
            0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
            4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 128), 7: (0, 128, 255),
            8: (128, 128, 0), 9: (255, 165, 0)
        }
        
        # Estatísticas
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        print(f"✓ TensorRT engine carregado com sucesso!")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")

    def setup_bindings(self):
        """Configurar bindings - versão sem warnings de deprecação"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        try:
            # Tentar usar APIs mais recentes
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                
                size = trt.volume(tensor_shape)
                dtype = trt.nptype(tensor_dtype)
                
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                    self.input_shape = tensor_shape
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    self.output_shape = tensor_shape
                    
        except AttributeError:
            # Fallback para APIs antigas (com warnings)
            print("Usando APIs antigas do TensorRT (com warnings)")
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(self.engine.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
                
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                    self.input_shape = self.engine.get_binding_shape(binding_idx)
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    self.output_shape = self.engine.get_binding_shape(binding_idx)

    def preprocess_frame(self, frame):
        """Preprocessar frame para TensorRT - CORRIGIDO"""
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(frame, (input_w, input_h))
        
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        # CORREÇÃO: usar np.ascontiguousarray() corretamente
        return np.ascontiguousarray(batched)

    def postprocess_output(self, output, original_shape):
        detections = []
        
        # Para YOLOv8 TensorRT, o formato pode ser [batch, num_classes + 4, num_anchors]
        # Ou [batch, num_anchors, num_classes + 4]
        
        if len(output.shape) == 3:
            # Reshape para [num_anchors, num_classes + 4]
            if output.shape[1] == 14:  # 4 coords + 10 classes
                output = output.transpose(0, 2, 1)  # [1, 2100, 14]
            output = output.reshape(-1, output.shape[-1])
        
        # Verificar se temos coordenadas válidas
        if output.shape[1] < 4:
            print(f"Output shape inválido: {output.shape}")
            return detections
        
        # Extrair coordenadas e scores
        boxes = output[:, :4]  # x, y, w, h
        scores = output[:, 4:]  # confidence + classes
        
        # Calcular confiança máxima por detecção
        if scores.shape[1] > 1:
            # Se temos classes separadas
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
        else:
            # Se só temos objectness
            max_scores = scores[:, 0]
            class_ids = np.zeros(len(scores), dtype=int)
        
        # Filtrar por confiança
        valid_mask = max_scores > self.confidence
        
        if not np.any(valid_mask):
            print(f"Nenhuma detecção acima do threshold {self.confidence}")
            return detections
        
        valid_boxes = boxes[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_classes = class_ids[valid_mask]
        
        # Converter coordenadas
        orig_h, orig_w = original_shape[:2]
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        for i, (box, score, class_id) in enumerate(zip(valid_boxes, valid_scores, valid_classes)):
            if class_id in self.target_classes:
                x_center, y_center, width, height = box
                
                # Converter para coordenadas absolutas
                x1 = int((x_center - width/2) * scale_x)
                y1 = int((y_center - height/2) * scale_y)
                x2 = int((x_center + width/2) * scale_x)
                y2 = int((y_center + height/2) * scale_y)
                
                # Validar coordenadas
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                if x2 > x1 and y2 > y1:  # Bounding box válida
                    detection_dict = {
                        'class_id': int(class_id),
                        'class_name': self.target_classes[class_id],
                        'confidence': float(score),
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection_dict)
        
        return detections


    def detect_frame(self, frame):
        """Detectar objetos em um frame usando TensorRT"""
        start_time = time.time()
        
        # Preprocessar
        input_data = self.preprocess_frame(frame)
        
        # Copiar dados para GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Executar inferência
        self.context.execute_v2(bindings=self.bindings)
        
        # Copiar resultado de volta
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Pós-processar
        output = self.outputs[0]['host'].reshape(self.output_shape)
        detections = self.postprocess_output(output, frame.shape)
        
        inference_time = time.time() - start_time
        
        # Anotar frame
        annotated_frame = self.annotate_frame(frame, detections)
        
        return detections, annotated_frame, inference_time

    def annotate_frame(self, frame, detections):
        """Anotar frame com detecções"""
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Desenhar bounding box
            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background do texto
            cv2.rectangle(annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color, -1)
            
            # Texto
            cv2.putText(annotated_frame, label,
                      (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (255, 255, 255), 2)
        
        return annotated_frame

    def calculate_fps(self):
        """Calcular FPS em tempo real"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return None

    def run_webcam(self, camera_id=0, save_snapshots=False):
        """Executar detecção em tempo real via webcam com TensorRT"""
        print(f"Iniciando webcam com TensorRT (ID: {camera_id})...")
        
        # Inicializar webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a webcam (ID: {camera_id})")
            return
        
        # Configurar resolução
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 TensorRT Webcam iniciada!")
        print("Controles:")
        print("  's' - Salvar snapshot")
        print("  'q' ou ESC - Sair")
        print("  'c' - Limpar estatísticas")
        
        fps_display = 0
        total_detections = 0
        snapshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame")
                    break
                
                self.frame_count += 1
                
                # Detectar com TensorRT
                detections, annotated_frame, inference_time = self.detect_frame(frame)
                total_detections += len(detections)
                
                # Calcular FPS
                current_fps = self.calculate_fps()
                if current_fps is not None:
                    fps_display = current_fps
                
                # Adicionar informações na tela
                info_y = 30
                cv2.putText(annotated_frame, f"TensorRT FPS: {fps_display:.1f}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Deteccoes: {len(detections)}", 
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Tempo: {inference_time*1000:.1f}ms", 
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Total: {total_detections}", 
                           (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar detecções encontradas
                if detections:
                    det_y = info_y + 120
                    for det in detections[:3]:
                        text = f"{det['class_name']}: {det['confidence']:.2f}"
                        cv2.putText(annotated_frame, text, 
                                   (10, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        det_y += 25
                
                # Mostrar frame
                cv2.imshow('TensorRT YOLO', annotated_frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s') and save_snapshots:
                    snapshot_count += 1
                    filename = f"tensorrt_snapshot_{snapshot_count:04d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Snapshot TensorRT salvo: {filename}")
                elif key == ord('c'):
                    total_detections = 0
                    self.frame_count = 0
                    print("Estatísticas limpas")
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n=== Estatísticas TensorRT ===")
            print(f"Frames processados: {self.frame_count}")
            print(f"Total de detecções: {total_detections}")
            print(f"Performance média: {fps_display:.1f} FPS")

def main():
    parser = argparse.ArgumentParser(description='TensorRT YOLO Webcam Inference')
    parser.add_argument('--engine', '-e', default='best.engine',
                       help='Caminho para engine TensorRT')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Threshold de confiança')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID da câmera')
    parser.add_argument('--save-snapshots', action='store_true',
                       help='Permitir salvar snapshots')
    
    args = parser.parse_args()
    
    # Criar detector TensorRT
    detector = TensorRTWebcamInference(args.engine, args.confidence)
    
    # Executar webcam
    detector.run_webcam(args.camera, args.save_snapshots)

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
 
        detector = TensorRTWebcamInference()
        detector.run_webcam()
    else:
        main()

