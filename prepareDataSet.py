import os
import random
import shutil

# Caminhos das pastas originais
images_dir = "dataset/images"
labels_dir = "dataset/labels"

# Caminhos das novas pastas
dest_dir = "dataset_split"
train_images_dir = os.path.join(dest_dir, "train", "images")
train_labels_dir = os.path.join(dest_dir, "train", "labels")
val_images_dir = os.path.join(dest_dir, "val", "images")
val_labels_dir = os.path.join(dest_dir, "val", "labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Define a proporção de treino/validação (ex: 8 = 80% treino, 20% val)
train_ratio = 0.8  

random.shuffle(image_files)

split_index = int(len(image_files) * train_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_files(files, images_src, labels_src, images_dst, labels_dst):
    for base_name in files:
        for ext in ['.jpg', '.jpeg', '.png']:
            src_image = os.path.join(images_src, base_name + ext)
            if os.path.exists(src_image):
                dst_image = os.path.join(images_dst, base_name + ext)
                shutil.copy2(src_image, dst_image)
                break

        src_label = os.path.join(labels_src, base_name + '.txt')
        if os.path.exists(src_label):
            dst_label = os.path.join(labels_dst, base_name + '.txt')
            shutil.copy2(src_label, dst_label)

copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
copy_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

print(f"Dataset dividido: {len(train_files)} treino, {len(val_files)} validação.")

