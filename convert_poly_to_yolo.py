import os
import glob
from pathlib import Path

# Diretório base do dataset
DATASET_DIR = "/home/ndo-vale/Desktop/Team06/datasets/roboflow_002"

# Pastas de labels a processar
SPLIT_DIRS = ["train", "valid", "test"]

def convert_polygons_to_yolo(input_label_dir, output_label_dir):
    """Converte anotações de polígonos para bounding boxes no formato YOLO."""
    # Criar diretório de saída
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)
    
    # Processar cada arquivo de label
    for label_path in glob.glob(f"{input_label_dir}/*.txt"):
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        output_lines = []
        for line in lines:
            values = list(map(float, line.strip().split()))
            if len(values) < 5:  # Mínimo: class + 4 valores (2 pontos x, y)
                print(f"Anotação inválida em {label_path}: {line.strip()}")
                continue
            
            class_idx = int(values[0])  # Índice da classe
            points = values[1:]  # Coordenadas x, y do polígono
            
            # Separar x e y (assumindo pares x, y)
            x_coords = points[0::2]  # x1, x2, x3, ...
            y_coords = points[1::2]  # y1, y2, y3, ...
            
            if len(x_coords) != len(y_coords):
                print(f"Coordenadas inválidas em {label_path}: {line.strip()}")
                continue
            
            # Calcular bounding box
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Adicionar ao formato YOLO
            output_lines.append(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        
        # Salvar arquivo convertido
        output_path = os.path.join(output_label_dir, os.path.basename(label_path))
        with open(output_path, "w") as f:
            f.writelines(output_lines)
        
        print(f"Convertido: {label_path} -> {output_path}")

def main():
    for split in SPLIT_DIRS:
        input_label_dir = os.path.join(DATASET_DIR, split, "labels")
        output_label_dir = os.path.join(DATASET_DIR, split, "labels_converted")
        
        if not os.path.exists(input_label_dir):
            print(f"Diretório não encontrado: {input_label_dir}")
            continue
        
        print(f"Processando {split}/labels...")
        convert_polygons_to_yolo(input_label_dir, output_label_dir)
        print(f"Conversão concluída para {split}. Arquivos salvos em {output_label_dir}")

if __name__ == "__main__":
    main()