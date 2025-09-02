import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os
class YOLOWebcamInference:
    def __init__(self, model_path='best.pt', confidence=0.8):
        print(f"Carregando modelo: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        self.target_classes = {
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
        
        #'person', 'car', 'stop sign', 'lane', 'passadeira', 'sinal verde', 'sinal amarelo', 'sinal vermelho', 'sinal perigo', 'speed 50', 'speed 80', 'jetracer', 'drivable', 'prioridade', 'gate closed'

    

        # # Cores para diferentes classes (BGR format)
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
        
        # Estatísticas
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

    def detect_frame(self, frame):
        """Detectar objetos em um frame"""
        start_time = time.time()
        
        # Fazer inferência
        results = self.model(frame, conf=self.confidence, verbose=False)
        inference_time = time.time() - start_time
        
        # Processar resultados
        detections = []
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Extrair informações
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filtrar apenas classes relevantes
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
                        
                        # Desenhar bounding box
                        color = self.colors.get(cls, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Desenhar label
                        label = f"{class_name}: {conf:.2f}"
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
        
        return detections, annotated_frame, inference_time

    def calculate_fps(self):
        """Calcular FPS em tempo real"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # A cada segundo
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return None

    def run_webcam(self, camera_id=0, save_snapshots=True):
        """Executar detecção em tempo real via webcam"""
        print(f"Iniciando webcam (ID: {camera_id})...")
        
        # Inicializar webcam
        cap = cv2.VideoCapture(camera_id)
        #cap = cv2.VideoCapture("http://100.93.94.123:8080/stream.mjpg")
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a webcam (ID: {camera_id})")
            return
        
        # Configurar resolução (opcional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam iniciada com sucesso!")
        print("Controles:")
        print("  's' - Salvar snapshot")
        print("  'q' ou ESC - Sair")
        print("  'c' - Limpar estatísticas")
        
        fps_display = 0
        total_detections = 0
        snapshot_count = 0
        
        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame da webcam")
                    break
                
                self.frame_count += 1
                
                # Detectar objetos
                detections, annotated_frame, inference_time = self.detect_frame(frame)
                total_detections += len(detections)
                
                # Calcular FPS
                current_fps = self.calculate_fps()
                if current_fps is not None:
                    fps_display = current_fps
                
                # Adicionar informações na tela
                info_y = 30
                cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", 
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
                    for det in detections[:3]:  # Mostrar apenas as 3 primeiras
                        text = f"{det['class_name']}: {det['confidence']:.2f}"
                        cv2.putText(annotated_frame, text, 
                                   (10, det_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        det_y += 25
                
                # Mostrar frame
                cv2.imshow('YOLO Webcam ', annotated_frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' ou ESC
                    break
                elif key == ord('s') and save_snapshots:  # Salvar snapshot
                    snapshot_count += 1
                    filename = f"snapshot_{snapshot_count:04d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Snapshot salvo: {filename}")
                elif key == ord('c'):  # Limpar estatísticas
                    total_detections = 0
                    self.frame_count = 0
                    print("Estatísticas limpas")
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        
        finally:
            # Limpeza
            cap.release()
            cv2.destroyAllWindows()
            
            # Estatísticas finais
            print(f"\n=== Estatísticas Finais ===")
            print(f"Frames processados: {self.frame_count}")
            print(f"Total de detecções: {total_detections}")
            if save_snapshots:
                print(f"Snapshots salvos: {snapshot_count}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Webcam Inference')
    parser.add_argument('--model', '-m', default='best.pt',
                       help='Caminho para modelo YOLO')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Threshold de confiança')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID da câmera (default: 0)')
    parser.add_argument('--save-snapshots', action='store_true',
                       help='Permitir salvar snapshots com tecla "s"')
    
    args = parser.parse_args()
    
 
    detector = YOLOWebcamInference(args.model, args.confidence)
    
 
    detector.run_webcam(args.camera, args.save_snapshots)

if __name__ == "__main__":
    if len(os.sys.argv) == 1:

        detector = YOLOWebcamInference()
        detector.run_webcam()
    else:
        main()

