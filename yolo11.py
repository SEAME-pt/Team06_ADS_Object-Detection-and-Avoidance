import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os
from pathlib import Path
from collections import deque

class TensorRTYOLO11:
    def __init__(self, engine_path, input_size=352):
        self.input_size = input_size
        self.engine_path = engine_path
        self.classes = {
            0: 'person',
            1: 'car',
            2: 'stop sign',
            3: 'passadeira',
            4: 'sinal verde',
            5: 'sinal amarelo',
            6: 'sinal vermelho',
            7: 'sinal perigo',
            8: 'speed 50',
            9: 'speed 80',
            10: 'jetracer',
            11: 'prioridade',
            12: 'gate closed'
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
            12: (255, 255, 0)
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
        
        # Para YOLO11, o formato é diferente - vamos detectar automaticamente
        output_size = self.outputs[0].host.shape[0]
        self.num_classes = len(self.classes)
        
        # YOLO11 tem formato [batch, 4+num_classes, num_anchors]
        # Calcular número de anchors baseado no tamanho total
        expected_channels = 4 + self.num_classes  # 4 coords + classes
        self.num_anchors = output_size // expected_channels
        
        print(f"Detectadas {self.num_classes} classes")
        print(f"Formato esperado: [1, {expected_channels}, {self.num_anchors}]")
        print(f"Tamanho total da saída: {output_size}")

    def postprocess_yolo11(self, output, scale, dw, dh, conf_threshold=0.25, nms_threshold=0.4):
        """Postprocessing específico para YOLO11"""
        try:
            # YOLO11 formato: [4+num_classes, num_anchors]
            expected_channels = 4 + self.num_classes
            
            # Verificar se o output tem o formato correto
            if len(output.shape) == 1:
                # Reshape para o formato esperado [channels, anchors]
                num_anchors = output.shape[0] // expected_channels
                output = output.reshape(expected_channels, num_anchors)
            
            # Transposicionar para [num_anchors, channels]
            output = output.T
            
            print(f"Shape após reshape: {output.shape}")
            
            # Separar coordenadas e scores
            boxes = output[:, :4]  # primeiras 4 colunas são coordenadas
            scores = output[:, 4:]  # restantes são scores das classes
            
            # Encontrar a classe com maior score para cada detecção
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)
            
            # Filtrar por threshold de confiança
            mask = max_scores > conf_threshold
            if not np.any(mask):
                return []
            
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            confidences = max_scores[mask]
            
            # Converter coordenadas do formato YOLO11 (center_x, center_y, width, height)
            # para formato de caixa (x1, y1, x2, y2)
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # Escalar de volta para a imagem original
            x_center = (x_center - dw) / scale
            y_center = (y_center - dh) / scale
            width = width / scale
            height = height / scale
            
            # Converter para x1, y1, x2, y2
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Preparar para NMS (formato: x, y, width, height)
            boxes_nms = np.column_stack([x1, y1, width, height])
            
            # Aplicar Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(),
                confidences.tolist(),
                conf_threshold,
                nms_threshold
            )
            
            detections = []
            if len(indices) > 0:
                # Garantir que indices é uma lista
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices[0]) > 0 else []
                elif isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                for i in indices:
                    detection = {
                        'class_id': int(class_ids[i]),
                        'class_name': self.classes.get(int(class_ids[i]), 'unknown'),
                        'confidence': float(confidences[i]),
                        'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Erro no postprocessing YOLO11: {e}")
            print(f"Shape do output: {output.shape if hasattr(output, 'shape') else 'N/A'}")
            return []

    def infer(self, image):
        """Inferência com debug adicional para YOLO11"""
        try:
            input_data, scale, dw, dh = self.preprocess(image)
            
            # Copiar dados para GPU
            np.copyto(self.inputs[0].host, input_data.ravel())
            
            # Transfer para GPU
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            
            # Executar inferência
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                print("Falha na execução da inferência")
                return []
            
            # Transfer de volta da GPU
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            
            # Sincronizar
            self.stream.synchronize()
            
            # Processar saída
            output = self.outputs[0].host
            
            # Debug: mostrar informações sobre a saída
            print(f"Output raw shape: {output.shape}, size: {output.size}")
            
            # Usar postprocessing específico para YOLO11
            detections = self.postprocess_yolo11(output, scale, dw, dh, conf_threshold=0.1, nms_threshold=0.4)
            
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
            
            # Alocar memória paginada no host
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream

    def preprocess(self, image):
        """Preprocessar imagem para inferência YOLO11"""
        h, w = image.shape[:2]
        
        # Calcular escala mantendo aspect ratio
        scale = min(self.input_size/w, self.input_size/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        
        # Criar imagem com padding
        image_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        # Normalizar e reformatar para CHW
        image_padded = image_padded.astype(np.float32) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))
        
        return image_padded, scale, dw, dh

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

class FPSCalculator:
    """Classe para calcular FPS de forma estável e precisa"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
        
    def update(self):
        """Atualiza o cálculo de FPS"""
        current_time = time.time()
        delta_time = current_time - self.last_time
        
        # Só adiciona se o delta time for razoável (evita valores extremos)
        if 0.001 < delta_time < 1.0:  # Entre 1ms e 1s
            self.frame_times.append(delta_time)
        
        self.last_time = current_time
        self.frame_count += 1
        
    def get_fps(self):
        """Retorna FPS médio baseado na janela de tempo"""
        if len(self.frame_times) < 2:
            return 0.0
        
        # Média dos tempos entre frames
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # Evitar divisão por zero
        if avg_frame_time <= 0:
            return 0.0
            
        return 1.0 / avg_frame_time
    
    def get_smooth_fps(self):
        """Retorna FPS suavizado removendo outliers"""
        if len(self.frame_times) < 5:
            return self.get_fps()
        
        # Remove os 20% maiores e menores valores (outliers)
        sorted_times = sorted(self.frame_times)
        trim_count = max(1, len(sorted_times) // 5)
        trimmed_times = sorted_times[trim_count:-trim_count]
        
        if not trimmed_times:
            return self.get_fps()
        
        avg_frame_time = sum(trimmed_times) / len(trimmed_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

class FrameSkipper:
    """Classe para gerenciar frame skipping com diferentes estratégias"""
    
    def __init__(self, skip_strategy="fixed", skip_frames=2, target_fps=15):
        self.skip_strategy = skip_strategy
        self.skip_frames = skip_frames
        self.target_fps = target_fps
        self.frame_counter = 0
        self.last_process_time = time.time()
        self.processing_times = deque(maxlen=20)  # Histórico limitado
        
    def should_process_frame(self):
        """Determina se o frame atual deve ser processado"""
        if self.skip_strategy == "fixed":
            should_process = (self.frame_counter % (self.skip_frames + 1)) == 0
        
        elif self.skip_strategy == "adaptive":
            current_time = time.time()
            if len(self.processing_times) > 5:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                target_interval = 1.0 / self.target_fps
                
                if avg_processing_time > target_interval:
                    skip_ratio = max(1, int(avg_processing_time / target_interval))
                    should_process = (self.frame_counter % skip_ratio) == 0
                else:
                    should_process = True
            else:
                should_process = True
        
        elif self.skip_strategy == "time_based":
            current_time = time.time()
            time_since_last = current_time - self.last_process_time
            target_interval = 1.0 / self.target_fps
            
            should_process = time_since_last >= target_interval
            
            if should_process:
                self.last_process_time = current_time
        
        else:
            should_process = True
        
        self.frame_counter += 1
        return should_process
    
    def record_processing_time(self, processing_time):
        """Registra tempo de processamento para estratégia adaptativa"""
        if 0.001 < processing_time < 5.0:
            self.processing_times.append(processing_time)

def setup_csi_camera(width=640, height=480, fps=30, flip_method=0):
    """Setup para câmera CSI (Jetson Nano/Xavier)"""
    gst_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1"
    )
    return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

def main():
    parser = argparse.ArgumentParser(description='TensorRT YOLO11 com Frame Skipping e FPS Corrigido')
    parser.add_argument('--engine', default='best.engine', help='Caminho para o arquivo engine')
    parser.add_argument('--input-size', type=int, default=352, help='Tamanho de entrada do modelo')
    parser.add_argument('--skip-strategy', choices=['fixed', 'adaptive', 'time_based'], 
                       default='fixed', help='Estratégia de frame skipping')
    parser.add_argument('--skip-frames', type=int, default=8, 
                       help='Número de frames para pular (estratégia fixed)')
    parser.add_argument('--target-fps', type=int, default=60, 
                       help='FPS alvo para estratégias adaptativa e time_based')
    parser.add_argument('--camera-source', type=int, default=0, 
                       help='Fonte da câmera (0 para webcam, ou URL para stream)')
    parser.add_argument('--use-csi', action='store_true', 
                       help='Usar câmera CSI (Jetson)')
    
    args = parser.parse_args()
    
    print("Inicializando TensorRT para YOLO11...")
    
    # Verificar CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU disponível: {torch.cuda.is_available()}")
            print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch não disponível, continuando...")
    
    # Inicializar detector
    detector = TensorRTYOLO11(args.engine, args.input_size)
    
    # Inicializar frame skipper
    frame_skipper = FrameSkipper(
        skip_strategy=args.skip_strategy,
        skip_frames=args.skip_frames,
        target_fps=args.target_fps
    )
    
    # Inicializar calculador de FPS
    fps_calculator = FPSCalculator(window_size=30)
    
    print(f"Configurando camera...")
    print(f"Estratégia de frame skipping: {args.skip_strategy}")
    if args.skip_strategy == "fixed":
        print(f"Pulando {args.skip_frames} frames entre processamentos")
    elif args.skip_strategy in ["adaptive", "time_based"]:
        print(f"FPS alvo: {args.target_fps}")
    
    # Configurar câmera
    if args.use_csi:
        cap = setup_csi_camera()
        print("Usando câmera CSI")
    else:
        cap = cv2.VideoCapture(args.camera_source)
        print(f"Usando câmera USB/webcam: {args.camera_source}")
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a camera")
        return
    
    print("Camera inicializada com sucesso!")
    print("Pressione 'q' para sair")
    print("Pressione '1', '2', '3' para mudar estratégia de frame skipping")
    
    # Variáveis de controle
    frame_count = 0
    processed_frames = 0
    skipped_frames = 0
    last_detections = []
    
    # Estatísticas
    stats_interval = 5.0  # Mostrar stats a cada 5 segundos
    last_stats_time = time.time()
    stats_frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Atualizar FPS
            fps_calculator.update()
            
            frame_count += 1
            stats_frame_count += 1
            
            # Decidir se processa o frame
            if frame_skipper.should_process_frame():
                # Processar frame
                inference_start = time.time()
                detections = detector.infer(frame)
                inference_time = time.time() - inference_start
                
                # Registrar tempo de processamento
                frame_skipper.record_processing_time(inference_time)
                
                processed_frames += 1
                last_detections = detections
                
                print(f"Frame {frame_count}: {len(detections)} detecções em {inference_time*1000:.1f}ms")
            else:
                # Usar últimas detecções
                detections = last_detections
                skipped_frames += 1
            
            # Desenhar detecções
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_id = det['class_id']
                confidence = det['confidence']
                class_name = det['class_name']
                
                color = detector.colors.get(class_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Mostrar estatísticas periódicas
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                elapsed = current_time - last_stats_time
                
                # Evitar divisão por zero
                period_fps = stats_frame_count / elapsed if elapsed > 0 else 0
                processing_ratio = processed_frames / frame_count * 100 if frame_count > 0 else 0
                
                print(f"Stats (últimos {stats_interval}s): FPS={period_fps:.1f} | "
                      f"Processados: {processed_frames}/{frame_count} ({processing_ratio:.1f}%) | "
                      f"Saltados: {skipped_frames}")
                
                last_stats_time = current_time
                stats_frame_count = 0
            
            # Overlay de informações
            smooth_fps = fps_calculator.get_smooth_fps()
            info_text = f"FPS: {smooth_fps:.1f} | Proc: {processed_frames} | Skip: {skipped_frames}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            strategy_text = f"Estrategia: {args.skip_strategy}"
            cv2.putText(frame, strategy_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            detections_text = f"Deteccoes: {len(detections)}"
            cv2.putText(frame, detections_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow('TensorRT YOLO11 - Frame Skipping', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                frame_skipper.skip_strategy = "fixed"
                print("Mudou para estratégia: fixed")
            elif key == ord('2'):
                frame_skipper.skip_strategy = "adaptive"
                print("Mudou para estratégia: adaptive")
            elif key == ord('3'):
                frame_skipper.skip_strategy = "time_based"
                print("Mudou para estratégia: time_based")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        final_fps = fps_calculator.get_smooth_fps()
        
        print(f"\n=== Estatísticas Finais ===")
        print(f"Total de frames capturados: {frame_count}")
        print(f"Frames processados: {processed_frames}")
        print(f"Frames saltados: {skipped_frames}")
        
        # Evitar divisão por zero
        if frame_count > 0:
            processing_ratio = processed_frames / frame_count * 100
            print(f"Taxa de processamento: {processing_ratio:.1f}%")
        else:
            print("Taxa de processamento: N/A")
            
        print(f"FPS final (suavizado): {final_fps:.1f}")

if __name__ == "__main__":
    main()
