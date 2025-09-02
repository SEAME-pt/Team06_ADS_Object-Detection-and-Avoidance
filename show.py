#!/usr/bin/env python3
"""
Visualizador de Labels YOLO com Pygame
Mostra imagens com bounding boxes e segmentação sobrepostas
"""

import pygame
import sys
import os
from pathlib import Path
import yaml
import cv2
import numpy as np

class YOLOViewer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.current_image_index = 0
        self.images = []
        self.labels = []
        self.class_names = []
        self.colors = []
        
        # Inicializa pygame
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("YOLO Label Viewer - Use ← → para navegar, ESC para sair")
        
        # Carrega dados
        self.load_dataset_info()
        self.load_images_and_labels()
        self.generate_colors()
        
        # Font para texto
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Estados para confirmação de delete
        self.delete_confirmation = False
        self.confirmation_timer = 0
    
    def load_dataset_info(self):
        """Carrega informações das classes"""
        # Mapeamento direto dos nomes das classes
        CLASS_NAMES = {
            0: 'STOP', 1: 'PASSAGEM', 2: 'VEL_50', 3: 'VEL_80',
            4: 'SEMAFORO_VERMELHO', 5: 'SEMAFORO_VERDE', 
            6: 'SEMAFORO_LARANJA', 7: 'PASSADEIRA', 8: 'DANGER', 9: 'CURVA'
        }
        
        # Converte para lista ordenada por ID
        max_id = max(CLASS_NAMES.keys()) if CLASS_NAMES else 0
        self.class_names = ['unknown'] * (max_id + 1)
        
        for class_id, class_name in CLASS_NAMES.items():
            self.class_names[class_id] = class_name
        
        print(f"Classes carregadas: {len(CLASS_NAMES)} classes")
        for class_id, class_name in CLASS_NAMES.items():
            print(f"  {class_id}: {class_name}")
    
    def load_images_and_labels(self):
        """Carrega lista de imagens e labels"""
        # Procura imagens em train e val
        image_dirs = [
            os.path.join(self.dataset_path, 'images', 'train'),
            os.path.join(self.dataset_path, 'images', 'val')
        ]
        
        label_dirs = [
            os.path.join(self.dataset_path, 'labels', 'train'),
            os.path.join(self.dataset_path, 'labels', 'val')
        ]
        
        # Extensões de imagem suportadas
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for img_dir, lbl_dir in zip(image_dirs, label_dirs):
            if os.path.exists(img_dir):
                for img_path in Path(img_dir).iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        # Procura label correspondente
                        label_path = Path(lbl_dir) / (img_path.stem + '.txt')
                        
                        self.images.append(str(img_path))
                        self.labels.append(str(label_path) if label_path.exists() else None)
        
        print(f"Encontradas {len(self.images)} imagens")
    
    def generate_colors(self):
        """Gera cores para cada classe"""
        np.random.seed(42)  # Para cores consistentes
        self.colors = []
        for i in range(len(self.class_names)):
            color = tuple(np.random.randint(0, 255, 3))
            self.colors.append(color)
    
    def load_labels(self, label_path):
        """Carrega labels de um arquivo"""
        if not label_path or not os.path.exists(label_path):
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        labels.append({'class_id': class_id, 'coords': coords})
        return labels
    
    def draw_bounding_box(self, surface, x, y, w, h, color, class_name, confidence=None):
        """Desenha bounding box"""
        # Converte coordenadas normalizadas para pixels
        img_h, img_w = surface.get_height(), surface.get_width()
        
        center_x = int(x * img_w)
        center_y = int(y * img_h)
        box_w = int(w * img_w)
        box_h = int(h * img_h)
        
        # Coordenadas do retângulo
        left = center_x - box_w // 2
        top = center_y - box_h // 2
        
        # Desenha retângulo
        pygame.draw.rect(surface, color, (left, top, box_w, box_h), 2)
        
        # Desenha label
        label_text = class_name
        if confidence:
            label_text += f" {confidence:.2f}"
        
        text_surface = self.small_font.render(label_text, True, color)
        text_bg = pygame.Rect(left, top - 20, text_surface.get_width() + 4, 20)
        pygame.draw.rect(surface, (0, 0, 0), text_bg)
        surface.blit(text_surface, (left + 2, top - 18))
    
    def delete_current_files(self):
        """Deleta a imagem atual e seu arquivo de label correspondente"""
        if not self.images:
            return False
            
        current_image_path = self.images[self.current_image_index]
        current_label_path = self.labels[self.current_image_index]
        
        deleted_files = []
        
        try:
            # Deleta imagem
            if os.path.exists(current_image_path):
                os.remove(current_image_path)
                deleted_files.append(os.path.basename(current_image_path))
            
            # Deleta label se existir
            if current_label_path and os.path.exists(current_label_path):
                os.remove(current_label_path)
                deleted_files.append(os.path.basename(current_label_path))
            
            # Remove da lista
            self.images.pop(self.current_image_index)
            self.labels.pop(self.current_image_index)
            
            # Ajusta índice atual
            if self.current_image_index >= len(self.images):
                self.current_image_index = max(0, len(self.images) - 1)
            
            print(f"Arquivos deletados: {', '.join(deleted_files)}")
            return True
            
        except Exception as e:
            print(f"Erro ao deletar arquivos: {e}")
            return False
        """Desenha polígono de segmentação"""
        if len(coords) < 6:  # Precisa de pelo menos 3 pontos (6 coordenadas)
            return
        
        img_h, img_w = surface.get_height(), surface.get_width()
        
        # Converte coordenadas normalizadas para pixels
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i + 1] * img_h)
            points.append((x, y))
        
        if len(points) >= 3:
            # Desenha polígono preenchido com transparência
            temp_surface = pygame.Surface((img_w, img_h), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surface, (*color, 80), points)
            surface.blit(temp_surface, (0, 0))
            
            # Desenha contorno
            pygame.draw.polygon(surface, color, points, 2)
    def draw_segmentation(self, surface, coords, color):
        """Desenha polígono de segmentação"""
        if len(coords) < 6:  # Precisa de pelo menos 3 pontos (6 coordenadas)
            return
        
        img_h, img_w = surface.get_height(), surface.get_width()
        
        # Converte coordenadas normalizadas para pixels
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i + 1] * img_h)
            points.append((x, y))
        
        if len(points) >= 3:
            # Desenha polígono preenchido com transparência
            temp_surface = pygame.Surface((img_w, img_h), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surface, (*color, 80), points)
            surface.blit(temp_surface, (0, 0))
            
            # Desenha contorno
            pygame.draw.polygon(surface, color, points, 2)
    def load_and_scale_image(self, image_path):
        """Carrega e redimensiona imagem para caber na tela"""
        # Carrega imagem com OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Converte BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calcula escala para caber na tela (deixando espaço para UI)
        max_width = self.screen_width - 50
        max_height = self.screen_height - 100
        
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensiona
        img = cv2.resize(img, (new_w, new_h))
        
        # Converte para surface do pygame
        surface = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        
        return surface
    
    def draw_ui(self):
        """Desenha interface do usuário"""
        # Fundo para informações
        info_height = 120 if self.delete_confirmation else 80
        info_rect = pygame.Rect(10, 10, 500, info_height)
        pygame.draw.rect(self.screen, (0, 0, 0), info_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), info_rect, 2)
        
        # Informações básicas
        if self.images:
            img_name = os.path.basename(self.images[self.current_image_index])
            info_text = [
                f"Imagem: {self.current_image_index + 1}/{len(self.images)}",
                f"Arquivo: {img_name}",
                f"Classes: {len([name for name in self.class_names if name != 'unknown'])}",
                "← → navegar | DEL deletar | ESC sair"
            ]
        else:
            info_text = ["Nenhuma imagem encontrada!", "", "", "ESC para sair"]
        
        for i, text in enumerate(info_text):
            color = (255, 255, 255)
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (15, 15 + i * 18))
        
        # Confirmação de delete
        if self.delete_confirmation:
            delete_text = [
                "⚠️  CONFIRMAR EXCLUSÃO:",
                "Pressione DEL novamente para confirmar",
                "Qualquer outra tecla para cancelar"
            ]
            
            for i, text in enumerate(delete_text):
                color = (255, 100, 100) if i == 0 else (255, 200, 200)
                text_surface = self.small_font.render(text, True, color)
                self.screen.blit(text_surface, (15, 87 + i * 16))
    
    def run(self):
        """Loop principal"""
        clock = pygame.time.Clock()
        running = True
        
        while running and self.images:
            current_time = pygame.time.get_ticks()
            
            # Reset confirmação de delete após 3 segundos
            if self.delete_confirmation and current_time - self.confirmation_timer > 3000:
                self.delete_confirmation = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_LEFT:
                        self.delete_confirmation = False
                        self.current_image_index = (self.current_image_index - 1) % len(self.images)
                    elif event.key == pygame.K_RIGHT:
                        self.delete_confirmation = False
                        self.current_image_index = (self.current_image_index + 1) % len(self.images)
                    elif event.key == pygame.K_DELETE:
                        if self.delete_confirmation:
                            # Confirma e deleta
                            success = self.delete_current_files()
                            self.delete_confirmation = False
                            if success and not self.images:
                                print("Todas as imagens foram deletadas!")
                                running = False
                        else:
                            # Inicia confirmação
                            self.delete_confirmation = True
                            self.confirmation_timer = current_time
                    else:
                        # Qualquer outra tecla cancela a confirmação
                        if self.delete_confirmation:
                            self.delete_confirmation = False
            
            # Limpa tela
            self.screen.fill((50, 50, 50))
            
            # Carrega imagem atual
            if self.images:  # Verifica se ainda há imagens
                current_image_path = self.images[self.current_image_index]
                current_label_path = self.labels[self.current_image_index]
            
            image_surface = self.load_and_scale_image(current_image_path)
            
            if image_surface:
                # Centraliza imagem
                img_rect = image_surface.get_rect()
                img_rect.center = (self.screen_width // 2, (self.screen_height + 100) // 2)
                
                # Desenha imagem
                self.screen.blit(image_surface, img_rect)
                
                # Carrega e desenha labels
                labels = self.load_labels(current_label_path)
                
                for label in labels:
                    class_id = label['class_id']
                    coords = label['coords']
                    
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id] 
                        if class_name != 'unknown':  # Só desenha se a classe for válida
                            color = self.colors[class_id % len(self.colors)]
                    
                        if len(coords) == 4:
                            # Bounding box format: x_center, y_center, width, height
                            x, y, w, h = coords
                            
                            # Ajusta coordenadas para a imagem redimensionada
                            scaled_surface = pygame.Surface(img_rect.size, pygame.SRCALPHA)
                            self.draw_bounding_box(scaled_surface, x, y, w, h, color, class_name)
                            self.screen.blit(scaled_surface, img_rect)
                        
                        elif len(coords) > 4:
                            # Segmentation format: x1, y1, x2, y2, ...
                            scaled_surface = pygame.Surface(img_rect.size, pygame.SRCALPHA)
                            self.draw_segmentation(scaled_surface, coords, color)
                            
                            # Adiciona label no primeiro ponto
                            if len(coords) >= 2:
                                img_h, img_w = scaled_surface.get_height(), scaled_surface.get_width()
                                x = int(coords[0] * img_w) + img_rect.x
                                y = int(coords[1] * img_h) + img_rect.y
                                
                                text_surface = self.small_font.render(class_name, True, color)
                                text_bg = pygame.Rect(x, y - 20, text_surface.get_width() + 4, 20)
                                pygame.draw.rect(self.screen, (0, 0, 0), text_bg)
                                self.screen.blit(text_surface, (x + 2, y - 18))
                            
                            self.screen.blit(scaled_surface, img_rect)
            
            # Desenha UI
            self.draw_ui()
            
            # Atualiza tela
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/home/djoker/code/yolo2/dataset"  # Caminho padrão
    
    if not os.path.exists(dataset_path):
        print(f"Erro: Dataset não encontrado em {dataset_path}")
        print("Uso: python label_viewer.py [caminho_dataset]")
        return
    
    print("Iniciando visualizador...")
    print("Controles:")
    print("  ← → : Navegar entre imagens")
    print("  DEL : Deletar imagem e label atual (com confirmação)")
    print("  ESC : Sair")
    
    viewer = YOLOViewer(dataset_path)
    viewer.run()

if __name__ == "__main__":
    main()
