import json
import os
import cv2
import shutil
from pathlib import Path

# Configurações
DATASET_DIR = "TT100K"  # Diretório raiz do dataset
OUTPUT_DIR = "TT100K_yolo"  # Diretório de saída
TARGET_CLASSES = ["pne", "p26"]  # NoEntry, stop-sign
IMG_SIZE = (416, 416)  # Tamanho de redimensionamento (None para manter original)
SPLIT = "train"  # Ou "test" para processar o conjunto de teste

# Mapeamento de classes para índices
CLASS_MAP = {"pne": 0, "p26": 1}
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
        x, y, w, h = box
        x = (x * ratio) + pad_w
        y = (y * ratio) + pad_h
        w = w * ratio
        h = h * ratio
        new_boxes.append([x, y, w, h])
    
    return img_padded, new_boxes

def convert_to_yolo_format(boxes, img_width, img_height):
    """Converte bounding boxes para formato YOLO (normalizado)."""
    yolo_boxes = []
    for box in boxes:
        x, y, w, h = box
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        yolo_boxes.append([x_center, y_center, w_norm, h_norm])
    return yolo_boxes

def process_tt100k(split):
    # Criar diretórios de saída
    output_img_dir = Path(OUTPUT_DIR) / split / "images"
    output_label_dir = Path(OUTPUT_DIR) / split / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar anotações JSON
    with open(f"{DATASET_DIR}/annotations/{split}.json", "r") as f:
        annotations = json.load(f)
    
    # Filtrar imagens com classes desejadas
    selected_images = []
    for img_id, img_data in annotations["imgs"].items():
        objects = img_data["objects"]
        if any(obj["category"] in TARGET_CLASSES for obj in objects):
            selected_images.append(img_data)
    
    print(f"Imagens com sinais pne/p26 em {split}: {len(selected_images)}")
    
    # Processar cada imagem
    for img_data in selected_images:
        img_id = img_data["path"].split("/")[-1]
        img_path = f"{DATASET_DIR}/{split}/{img_id}"
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
        for obj in img_data["objects"]:
            if obj["category"] in TARGET_CLASSES:
                bbox = obj["bbox"]
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["w"]
                h = bbox["h"]
                boxes.append([x, y, w, h])
                class_ids.append(CLASS_MAP[obj["category"]])
        
        # Redimensionar imagem e ajustar caixas
        img_processed, boxes = resize_image_and_boxes(img, boxes, IMG_SIZE)
        new_height, new_width = img_processed.shape[:2]
        
        # Converter para formato YOLO
        yolo_boxes = convert_to_yolo_format(boxes, new_width, new_height)
        
        # Salvar imagem
        output_img_path = output_img_dir / img_id
        cv2.imwrite(str(output_img_path), img_processed)
        
        # Salvar anotações
        output_label_path = output_label_dir / f"{img_id.split('.')[0]}.txt"
        with open(output_label_path, "w") as f:
            for class_id, box in zip(class_ids, yolo_boxes):
                f.write(f"{class_id} {' '.join(map(str, box))}\n")
    
    # Criar data.yaml
    data_yaml = {
        "train": str(Path(OUTPUT_DIR) / "train" / "images"),
        "val": str(Path(OUTPUT_DIR) / "test" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        json.dump(data_yaml, f, indent=2)

def main():
    for split in ["train", "test"]:
        process_tt100k(split)
    print(f"Dataset preparado em: {OUTPUT_DIR}")
    print(f"Use o arquivo {OUTPUT_DIR}/data.yaml para treinar o YOLOv5.")

if __name__ == "__main__":
    main()