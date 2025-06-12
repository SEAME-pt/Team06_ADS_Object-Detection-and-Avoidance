import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os
from pathlib import Path

class ONNXYOLO:
    def __init__(self, onnx_path, input_size=320):
        self.input_size = input_size
        self.onnx_path = onnx_path
        
 
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
            0: (255, 0, 0),      # STOP - Azul
            1: (0, 255, 0),      # PASSAGEM - Verde
            2: (0, 0, 255),      # VEL_50 - Vermelho
            3: (255, 255, 0),    # VEL_80 - Ciano
            4: (255, 0, 255),    # SEMAFORO_VERMELHO - Magenta
            5: (0, 255, 255),    # SEMAFORO_VERDE - Amarelo
            6: (128, 0, 128),    # SEMAFORO_LARANJA - Roxo
            7: (0, 128, 255),    # PASSADEIRA - Laranja
            8: (128, 128, 0),    # DANGER - Verde escuro
            9: (255, 165, 0)     # CURVA - Azul claro
        }
        
        # Inicializar ONNX Runtime
        self.session = self.load_onnx_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Obter informações do modelo
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        print(f"ONNX Model carregado: {onnx_path}")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        
        # Detectar input_size do modelo se dinâmico
        if len(input_shape) >= 3:
            if input_shape[2] != -1:  # Se não for dinâmico
                self.input_size = input_shape[2]
            if input_shape[3] != -1:  # Se não for dinâmico
                self.input_size = input_shape[3]

    def load_onnx_model(self):
        """Carregar modelo ONNX"""
        try:
            # Configurar providers (GPU se disponível)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(self.onnx_path, providers=providers)
            
            # Verificar qual provider está sendo usado
            provider_used = session.get_providers()[0]
            print(f"ONNX Provider: {provider_used}")
            
            return session
        except Exception as e:
            print(f"Erro ao carregar modelo ONNX: {e}")
            raise

    def preprocess(self, image):
        """Preprocessar imagem para inferência - mesmo método do TensorRT"""
        h, w = image.shape[:2]
        
        # Resize mantendo aspect ratio
        scale = min(self.input_size/w, self.input_size/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        
        # Padding
        image_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        # Normalizar e converter para formato ONNX
        image_padded = image_padded.astype(np.float32) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
        image_padded = np.expand_dims(image_padded, axis=0)   # Add batch dimension
        
        return image_padded, scale, dw, dh

    def postprocess(self, output, scale, dw, dh, conf_threshold=0.5, nms_threshold=0.4):
        """Postprocessar saída ONNX YOLOv8 - FORMATO CORRIGIDO"""
        detections = []
        
        # YOLOv8 ONNX tem formato (1, 84, 8400) - precisa transpor
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            print(f"Original shape: {output.shape}")
            output = output.transpose(0, 2, 1)  # (1, 84, 8400) -> (1, 8400, 84)
            print(f"Transposed shape: {output.shape}")
        
        # Remover batch dimension
        output = output[0]  # (8400, 84)
        
        print(f"Processing {output.shape[0]} detections with {output.shape[1]} features")
        
        for detection in output:
            # YOLOv8: [x, y, w, h, class0_score, class1_score, ..., classN_score]
            # NÃO tem confidence separada - a confidence é o max das classes
            
            if len(detection) >= 5:  # Pelo menos x,y,w,h + 1 classe
                x_center, y_center, width, height = detection[:4]
                class_scores = detection[4:]  # Todas as classes
                
                # Encontrar classe com maior score (isso É a confidence)
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]  # Max score = confidence
                
                if confidence > conf_threshold and class_id in self.classes:
                    # Converter coordenadas para imagem original
                    x_center = (x_center - dw) / scale
                    y_center = (y_center - dh) / scale
                    width = width / scale
                    height = height / scale
                    
                    # Converter para x1, y1, x2, y2
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Validar coordenadas
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.classes[class_id],
                            'confidence': float(confidence),
                            'bbox': [x1, y1, x2, y2]
                        })
        
        print(f"Found {len(detections)} valid detections")
        
        # Aplicar NMS
        if len(detections) > 1:
            detections = self.apply_nms(detections, nms_threshold)
            print(f"After NMS: {len(detections)} detections")
        
        return detections


    def apply_nms(self, detections, nms_threshold):
        """Aplicar Non-Maximum Suppression - mesmo método do TensorRT"""
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
        """Executar inferência ONNX"""
        # Preprocessar
        input_data, scale, dw, dh = self.preprocess(image)
        
        # Inferência ONNX
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        # Postprocessar
        output = outputs[0]
        detections = self.postprocess(output, scale, dw, dh)
        
        return detections, inference_time

def setup_csi_camera(width=640, height=480, fps=30, flip_method=0):
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
    parser = argparse.ArgumentParser(description='ONNX YOLO Inference')
    parser.add_argument('--model', '-m', default='best.onnx', help='Caminho para modelo ONNX')
    parser.add_argument('--camera', '-c', type=int, default=0, help='ID da câmera')
    parser.add_argument('--csi', action='store_true', help='Usar câmera CSI')
    parser.add_argument('--stream', '-s', help='URL do stream MJPEG')
    
    args = parser.parse_args()
    
    print("Inicializando ONNX YOLO...")
    detector = ONNXYOLO(args.model, 320)
    
    # Configurar câmera
    if args.stream:
        print(f"Conectando ao stream: {args.stream}")
        cap = cv2.VideoCapture(args.stream)
    elif args.csi:
        print("Configurando camera CSI...")
        cap = setup_csi_camera(640, 480, 30, 0)
    else:
        print(f"Configurando webcam (ID: {args.camera})...")
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return
    
    print("Câmera inicializada com sucesso!")
    print("Pressione 'q' para sair")
    
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    total_inference_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Inferência ONNX
            detections, inference_time = detector.infer(frame)
            total_inference_time += inference_time
            
            # Desenhar detecções
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
            
            # Calcular FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = 30 / elapsed
                avg_inference = (total_inference_time / fps_counter) * 1000
                start_time = time.time()
                print(f"FPS: {current_fps:.1f} | Avg Inference: {avg_inference:.1f}ms | Detections: {len(detections)}")
            
            # Informações na tela
            current_fps = fps_counter / (time.time() - start_time + 0.001)
            info_text = f"ONNX FPS: {current_fps:.1f} | Detections: {len(detections)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('ONNX YOLO - Camera', frame)
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrompido")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_inference = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
        
        print(f"\n=== Estatísticas Finais ONNX ===")
        print(f"Total de frames processados: {frame_count}")
        print(f"FPS médio: {avg_fps:.1f}")
        print(f"Tempo médio de inferência: {avg_inference:.1f}ms")

if __name__ == "__main__":
    main()
# Teste básico com webcam
#python test_onnx_model.py

# Com modelo específico
#python test_onnx_model.py --model best.onnx --camera 0

# Com câmera CSI
#python test_onnx_model.py --model best.onnx --csi

# Com stream MJPEG
#python test_onnx_model.py --model best.onnx --stream "http://100.93.94.123:8080/stream.mjpg"

