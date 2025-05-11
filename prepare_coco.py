import json
import os
import cv2
import glob
from pathlib import Path
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import numpy as np

# Configurações
DATASET_DIR = "/home/ndo-vale/Desktop/Team06/datasets/coco2017"  # Ajuste para o caminho do COCO
OUTPUT_DIR = "/home/ndo-vale/Desktop/Team06/datasets/COCO_yolo"  # Diretório de saída
IMG_SIZE = None  # Manter tamanho original (None) ou definir como (416, 416)
TRAIN_SPLIT = 0.8  # Proporção para treino (80% train, 20% valid do train2017)

# Classes desejadas
TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat"
]
CLASS_MAP = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

# Função para redimensionar imagem e ajustar caixas
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

# Converter para formato YOLO
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

# Processar o dataset COCO
def process_coco():
    # Inicializar COCO para train e val
    train_ann_file = f"{DATASET_DIR}/annotations/instances_train2017.json"
    val_ann_file = f"{DATASET_DIR}/annotations/instances_val2017.json"
    
    if not os.path.exists(train_ann_file) or not os.path.exists(val_ann_file):
        print(f"Anotações não encontradas em: {DATASET_DIR}/annotations")
        return
    
    # Carregar COCO
    coco_train = COCO(train_ann_file)
    coco_val = COCO(val_ann_file)
    
    # Obter IDs das categorias desejadas
    cat_ids = coco_train.getCatIds(catNms=TARGET_CLASSES)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_train.loadCats(cat_ids)}
    
    # Função para processar um split
    def process_split(coco, img_ids, split_name, dataset_dir):
        output_img_dir = Path(OUTPUT_DIR) / split_name / "images"
        output_label_dir = Path(OUTPUT_DIR) / split_name / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = f"{dataset_dir}/{img_info['file_name']}"
            if not os.path.exists(img_path):
                print(f"Imagem não encontrada: {img_path}")
                continue
            
            # Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"Falha ao carregar imagem: {img_path}")
                continue
            img_height, img_width = img.shape[:2]
            
            # Carregar anotações
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            if not anns:
                continue
            
            boxes = []
            class_ids = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                cat_name = cat_id_to_name[ann["category_id"]]
                if cat_name not in TARGET_CLASSES:
                    continue
                x_min, y_min, w, h = ann["bbox"]
                x = x_min
                y = y_min
                boxes.append([x, y, w, h])
                class_ids.append(CLASS_MAP[cat_name])
            
            if not boxes:
                continue
            
            # Redimensionar imagem e ajustar caixas
            img_processed, boxes = resize_image_and_boxes(img, boxes, IMG_SIZE)
            new_height, new_width = img_processed.shape[:2]
            
            # Converter para formato YOLO
            yolo_boxes = convert_to_yolo_format(boxes, new_width, new_height)
            
            # Salvar imagem
            output_img_path = output_img_dir / img_info["file_name"]
            cv2.imwrite(str(output_img_path), img_processed)
            
            # Salvar anotações
            output_label_path = output_label_dir / f"{Path(img_info['file_name']).stem}.txt"
            with open(output_label_path, "w") as f:
                for class_id, box in zip(class_ids, yolo_boxes):
                    f.write(f"{class_id} {' '.join(map(str, box))}\n")
        
        print(f"Processado {split_name}: {len(img_ids)} imagens com classes {', '.join(TARGET_CLASSES)}")
    
    # Filtrar imagens de treino com classes desejadas
    train_img_ids = []
    for img_id in coco_train.getImgIds():
        ann_ids = coco_train.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        if ann_ids:
            train_img_ids.append(img_id)
    
    # Dividir train em train e valid
    train_img_ids, valid_img_ids = train_test_split(
        train_img_ids, train_size=TRAIN_SPLIT, random_state=42
    )
    
    print(f"Imagens de treino: {len(train_img_ids)}")
    print(f"Imagens de validação: {len(valid_img_ids)}")
    
    # Filtrar imagens de validação (usadas como test)
    test_img_ids = []
    for img_id in coco_val.getImgIds():
        ann_ids = coco_val.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        if ann_ids:
            test_img_ids.append(img_id)
    
    print(f"Imagens de teste: {len(test_img_ids)}")
    
    # Processar splits
    process_split(coco_train, train_img_ids, "train", f"{DATASET_DIR}/train2017")
    process_split(coco_train, valid_img_ids, "valid", f"{DATASET_DIR}/train2017")
    process_split(coco_val, test_img_ids, "test", f"{DATASET_DIR}/val2017")
    
    # Criar data.yaml
    data_yaml = {
        "train": str(Path(OUTPUT_DIR) / "train" / "images"),
        "val": str(Path(OUTPUT_DIR) / "valid" / "images"),
        "test": str(Path(OUTPUT_DIR) / "test" / "images"),
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        json.dump(data_yaml, f, indent=2)

def main():
    process_coco()
    print(f"Dataset preparado em: {OUTPUT_DIR}")
    print(f"Use o arquivo {OUTPUT_DIR}/data.yaml para treinar o YOLOv8.")

if __name__ == "__main__":
    main()