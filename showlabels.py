import pygame
import os
import glob
import sys

# Inicializa o Pygame
pygame.init()

# Configurações
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)

COLORS = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, CYAN, PINK]

# Nomes das classes (ajusta conforme as tuas classes)
CLASS_NAMES = [
    "STOP",
    "PASSAGEM", 
    "VEL_50",
    "VEL_80",
    "SEMAFORO_VERMELHO",
    "SEMAFORO_VERDE",
    "SEMAFORO_LARANJA",
    "PASSADEIRA",
    "DANGER",
    "CURVA" 


]


IMAGES_FOLDER = "dataset/images"
LABELS_FOLDER = "dataset/labels"

# Lista de imagens e labels
image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.png")))
if not image_files:
    print(f"Nenhuma imagem encontrada em {IMAGES_FOLDER}")
    sys.exit(1)

label_files = [os.path.join(LABELS_FOLDER, os.path.basename(f)[:-4] + ".txt") for f in image_files]

current_index = 0
show_labels = True
show_confidence = False  # Para futuras detecções com confiança
font_size = 24

# Cria a janela
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Visualizador de Bounding Boxes YOLO - Dataset Validator")

clock = pygame.time.Clock()
running = True

def validate_bbox(x_center, y_center, width, height):
    """Valida se a bounding box está dentro dos limites normalizados [0,1]"""
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        return False, "Centro fora dos limites [0,1]"
    if not (0 < width <= 1 and 0 < height <= 1):
        return False, "Dimensões inválidas"
    if (x_center - width/2 < 0 or x_center + width/2 > 1 or 
        y_center - height/2 < 0 or y_center + height/2 > 1):
        return False, "Box ultrapassa limites da imagem"
    return True, "OK"

def draw_bounding_boxes(surface, image_rect, label_path):
    """Desenha bounding boxes e retorna estatísticas"""
    if not os.path.exists(label_path):
        return {"total": 0, "valid": 0, "invalid": 0, "errors": []}
    
    stats = {"total": 0, "valid": 0, "invalid": 0, "errors": [], "boxes": []}
    
    try:
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    stats["errors"].append(f"Linha {line_num}: Formato inválido")
                    continue
                
                try:
                    # Substitui vírgula por ponto
                    corrected_parts = [part.replace(',', '.') for part in parts[:5]]
                    class_id, x_center, y_center, width, height = map(float, corrected_parts)
                    
                    stats["total"] += 1
                    
                    # Valida a bounding box
                    is_valid, error_msg = validate_bbox(x_center, y_center, width, height)
                    
                    if is_valid:
                        stats["valid"] += 1
                    else:
                        stats["invalid"] += 1
                        stats["errors"].append(f"Linha {line_num}: {error_msg}")
                    
                    # Converte para coordenadas absolutas
                    box_x = x_center * image_rect.w
                    box_y = y_center * image_rect.h
                    box_w = width * image_rect.w
                    box_h = height * image_rect.h
                    
                    # Calcula posição top-left
                    box_left = box_x - box_w / 2
                    box_top = box_y - box_h / 2
                    
                    # Escolhe cor baseada na validade
                    if is_valid:
                        color = COLORS[int(class_id) % len(COLORS)]
                    else:
                        color = RED
                    
                    # Desenha o retângulo
                    if show_labels:
                        pygame.draw.rect(surface, color, 
                                       (image_rect.x + box_left, image_rect.y + box_top, box_w, box_h), 2)
                        
                        # Desenha label da classe
                        class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class_{int(class_id)}"
                        font = pygame.font.SysFont(None, font_size)
                        text = font.render(class_name, True, color)
                        text_rect = text.get_rect()
                        text_rect.x = image_rect.x + box_left
                        text_rect.y = image_rect.y + box_top - text_rect.height - 2
                        
                        # Fundo para o texto
                        pygame.draw.rect(surface, WHITE, text_rect)
                        surface.blit(text, text_rect)
                    
                    # Guarda info da box para estatísticas
                    stats["boxes"].append({
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class_{int(class_id)}",
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "valid": is_valid
                    })
                    
                except ValueError as e:
                    stats["errors"].append(f"Linha {line_num}: Erro de conversão - {e}")
                    continue
                    
    except Exception as e:
        stats["errors"].append(f"Erro ao ler ficheiro: {e}")
    
    return stats

def draw_info_panel(surface, stats, current_img_name):
    """Desenha painel de informações"""
    panel_width = 300
    panel_x = WINDOW_WIDTH - panel_width
    
    # Fundo do painel
    pygame.draw.rect(surface, (240, 240, 240), (panel_x, 0, panel_width, WINDOW_HEIGHT))
    pygame.draw.line(surface, BLACK, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)
    
    font = pygame.font.SysFont(None, 20)
    font_small = pygame.font.SysFont(None, 16)
    y = 10
    line_height = 25
    
    # Informações gerais
    texts = [
        f"Imagem: {current_index + 1}/{len(image_files)}",
        f"Nome: {current_img_name}",
        "",
        f"Boxes Total: {stats['total']}",
        f"Boxes Válidas: {stats['valid']}",
        f"Boxes Inválidas: {stats['invalid']}",
        "",
        "Classes Detectadas:"
    ]
    
    for text in texts:
        if text:
            color = GREEN if "Válidas" in text else RED if "Inválidas" in text else BLACK
            text_surface = font.render(text, True, color)
            surface.blit(text_surface, (panel_x + 10, y))
        y += line_height
    
    # Lista de classes detectadas
    class_counts = {}
    for box in stats["boxes"]:
        class_name = box["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        text = f"  {class_name}: {count}"
        text_surface = font_small.render(text, True, BLACK)
        surface.blit(text_surface, (panel_x + 10, y))
        y += 20
    
    # Erros (se houver)
    if stats["errors"] and y < WINDOW_HEIGHT - 100:
        y += 10
        error_text = font.render("Erros:", True, RED)
        surface.blit(error_text, (panel_x + 10, y))
        y += line_height
        
        for error in stats["errors"][:5]:  # Máximo 5 erros
            error = error[:40] + "..." if len(error) > 40 else error
            error_surface = font_small.render(error, True, RED)
            surface.blit(error_surface, (panel_x + 10, y))
            y += 18
    
    # Controlos
    y = WINDOW_HEIGHT - 120
    controls = [
        "Controlos:",
        "← → Navegar",
        "L - Toggle Labels",
        "ESC - Sair",
        "R - Relatório"
    ]
    
    for control in controls:
        color = BLUE if control == "Controlos:" else BLACK
        text_surface = font_small.render(control, True, color)
        surface.blit(text_surface, (panel_x + 10, y))
        y += 18

def generate_report():
    """Gera relatório completo do dataset"""
    print("\n" + "="*60)
    print("RELATÓRIO DO DATASET")
    print("="*60)
    
    total_images = len(image_files)
    total_boxes = 0
    total_valid = 0
    total_invalid = 0
    class_distribution = {}
    
    for i, (img_file, label_file) in enumerate(zip(image_files, label_files)):
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            corrected_parts = [part.replace(',', '.') for part in parts[:5]]
                            class_id, x_center, y_center, width, height = map(float, corrected_parts)
                            total_boxes += 1
                            
                            is_valid, _ = validate_bbox(x_center, y_center, width, height)
                            if is_valid:
                                total_valid += 1
                            else:
                                total_invalid += 1
                            
                            class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class_{int(class_id)}"
                            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                        except:
                            total_invalid += 1
    
    print(f"Total de Imagens: {total_images}")
    print(f"Total de Bounding Boxes: {total_boxes}")
    print(f"Boxes Válidas: {total_valid} ({total_valid/total_boxes*100:.1f}%)" if total_boxes > 0 else "Boxes Válidas: 0")
    print(f"Boxes Inválidas: {total_invalid} ({total_invalid/total_boxes*100:.1f}%)" if total_boxes > 0 else "Boxes Inválidas: 0")
    print(f"Média de boxes por imagem: {total_boxes/total_images:.1f}" if total_images > 0 else "Média: 0")
    
    print("\nDistribuição por Classe:")
    for class_name, count in sorted(class_distribution.items()):
        percentage = count/total_boxes*100 if total_boxes > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("="*60)

# Loop principal
while running:
    screen.fill(WHITE)
    
    # Carrega e exibe a imagem atual
    stats = {"total": 0, "valid": 0, "invalid": 0, "errors": [], "boxes": []}
    
    if current_index < len(image_files):
        try:
            img = pygame.image.load(image_files[current_index])
            
            # Redimensiona mantendo proporção (deixa espaço para painel)
            available_width = WINDOW_WIDTH - 320
            img_rect = img.get_rect()
            scale = min(available_width / img_rect.w, WINDOW_HEIGHT / img_rect.h) * 0.9
            img_rect.w = int(img_rect.w * scale)
            img_rect.h = int(img_rect.h * scale)
            img_rect.center = (available_width // 2, WINDOW_HEIGHT // 2)
            
            img = pygame.transform.scale(img, (img_rect.w, img_rect.h))
            screen.blit(img, img_rect)
            
            # Desenha bounding boxes e obtém estatísticas
            stats = draw_bounding_boxes(screen, img_rect, label_files[current_index])
            
        except Exception as e:
            font = pygame.font.SysFont(None, 36)
            error_text = font.render(f"Erro ao carregar imagem: {e}", True, RED)
            screen.blit(error_text, (50, WINDOW_HEIGHT // 2))
    
    # Desenha painel de informações
    img_name = os.path.basename(image_files[current_index]) if current_index < len(image_files) else "N/A"
    draw_info_panel(screen, stats, img_name)
    
    pygame.display.flip()
    
    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                current_index = max(0, current_index - 1)
            elif event.key == pygame.K_RIGHT:
                current_index = min(len(image_files) - 1, current_index + 1)
            elif event.key == pygame.K_l:
                show_labels = not show_labels
            elif event.key == pygame.K_r:
                generate_report()
            elif event.key == pygame.K_ESCAPE:
                running = False
    
    clock.tick(60)

pygame.quit()
print("Visualizador fechado.")
