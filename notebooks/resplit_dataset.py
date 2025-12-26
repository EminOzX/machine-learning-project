import os
import shutil
import random

# ===================== AYARLAR =====================
DATASET_DIR = "dataset_split"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
# ==================================================

random.seed(RANDOM_SEED)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")
test_dir  = os.path.join(DATASET_DIR, "test")

def clear_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

print("Val ve Test klasörleri temizleniyor...")
clear_directory(val_dir)
clear_directory(test_dir)

print("Dataset yeniden bölünüyor...\n")

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)
    total = len(images)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)
    # kalan otomatik test
    n_test  = total - n_train - n_val

    val_class_dir = os.path.join(val_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # VAL
    for img in images[n_train:n_train + n_val]:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(val_class_dir, img)
        )

    # TEST
    for img in images[n_train + n_val:]:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(test_class_dir, img)
        )

    print(
        f"{class_name}: "
        f"Train={n_train}, Val={n_val}, Test={n_test}"
    )

print("\nBölme işlemi tamamlandı ✔")
