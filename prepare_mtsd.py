import json
import os
import cv2
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configurações
DATASET_DIR = "MTSD"  # Diretório raiz do dataset
OUTPUT_DIR = "MTSD_yolo"  # Diretório de saída
TARGET_CLASSES = ["regulatory--no-entry", "regulatory--stop"]  # NoEntry, stop-sign
IMG_SIZE = (416, 416)  # Tamanho de redimensionamento (None para manter original)
TRAIN_SPLIT = 0.8  # Proporção para treino

# Mapeamento de classes para índices
CLASS_MAP = {"regulatory--no-entry": 0, "regulatory--stop": 1}
CLASS_NAMES = ["NoEntry", "stop-sign"]

def resize_image_and_boxes(img, boxes, target_size):
    """Redimensiona a imagem e ajusta as bounding boxes."""
    h, w = img.shape[:2]
    if target_size is None:
        return img, boxes
    
    # Calcular proporção e padding
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Adicionar padding
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    
    # Ajustar bounding boxes
    new_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = (x_min * ratio) + pad_w
        y_min = (y_min * ratio) + pad_h
        x_max = (x_max * ratio) + pad_w
        y_max = (y_max * ratio) + pad_h
        new_boxes.append([x_min, y_min, x_max, y_max])
    
    return img_padded, new_boxes

def convert_to_yolo_format(boxes, img_width, img_height):
    """Converte bounding boxes para formato YOLO (normalizado)."""
    yolo_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        w_norm = (x_max - x_min) / img_width
        h_norm = (y_max - y_min) / img_height
        yolo_boxes.append([x_center, y_center, w_norm, h_norm])
    return yolo_boxes

def process_mtsd():
    # Carregar anotações JSON
    with open(f"{DATASET_DIR}/annotations.json", "r") as f:
        annotations = json.load(f)
    
    # Filtrar imagens com classes desejadas
    selected_images = []
    for img_data in annotations["images"]:
        img_id = img_data["id"]
        objects = [obj for obj in annotations["annotations"] if obj["image_id"] == img_id]
        if any(obj["category"] in TARGET_CLASSES for obj in objects):
            selected_images.append((img_data, objects))
    
    print(f"Imagens com sinais NoEntry/stop-sign: {len(selected_images)}")
    
    # Dividir em treino e validação
    train_images, val_images = train_test_split(selected_images, train_size=TRAIN_SPLIT, random_state=42)
    
    for split, split_images in [("train", train_images), ("val", val_images)]:
        # Criar diretórios de saída
        output_img_dir = Path(OUTPUT_DIR) / split / "images"
        output_label_dir = Path(OUTPUT_DIR) / split / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Processar cada imagem
        for img_data, objects in split_images:
            img_path = f"{DATASET_DIR}/images/{img_data['file_name']}"
            if not os.path.exists(img_path):
                print(f"Imagem não encontrada: {img_path}")
                continue
            
            # Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"Falha ao carregar imagem: {img_path}")
                continue
            img_height, img_width = img.shape[:2]
            
            # Extrair bounding boxes
            boxes = []
            class_ids = []
            for obj in objects:
                if obj["category"] in TARGET_CLASSES:
                    bbox = obj["bbox"]
                    x_min, y_min, x_max, y_max = bbox
                    boxes.append([x_min, y_min, x_max, y_max])
                    class_ids.append(CLASS_MAP[obj["category"]])
            
            # Redimensionar imagem e ajustar caixas
            img_processed, boxes = resize_image_and_boxes(img, boxes, IMG_SIZE)
            new_height, new_width = img_processed.shape[:2]
            
            # Converter para formato YOLO
            yolo_boxes = convert_to_yolo_format(boxes, new_width, new_height)
            
            # Salvar imagem
            output_img_path = output_img_dir / img_data["file_name"]
            cv2.imwrite(str(output_img_path), img_processed)
            
            # Salvar anotações
            output_label_path = output_label_dir / f"{img_data['file_name'].split('.')[0]}.txt"
            with open(output_label_path, "w") as f:
                for class_id, box in zip(class_ids, yolo_boxes):
                    f.write(f"{class_id} {' '.join(map(str, box))}\n")
    
    # Criar data.yaml
    data_yaml = {
        "train": str(Path(OUTPUT_DIR) / "train" / "images"),
        "val": str(Path(OUTPUT_DIR) / "val" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        json.dump(data_yaml, f, indent=2)

def main():
    process_mtsd()
    print(f"Dataset preparado em: {OUTPUT_DIR}")
    print(f"Use o arquivo {OUTPUT_DIR}/data.yaml para treinar o YOLOv5.")

if __name__ == "__main__":
    main()