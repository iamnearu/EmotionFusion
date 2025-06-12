# preprocess/split_dataset.py

import os
import csv
import random
import shutil

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Chia dữ liệu thành train/val/test và copy ảnh + văn bản vào thư mục tương ứng.
    """
    random.seed(seed)

    # Đọc toàn bộ mẫu từ label.csv
    label_path = os.path.join(data_dir, "label.csv")
    with open(label_path, encoding="utf-8") as f:
        next(f)  # bỏ header
        samples = [line.strip().split(',') for line in f]

    # Shuffle và chia dữ liệu
    random.shuffle(samples)
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    split_data = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    for split_name, split_samples in split_data.items():
        split_dir = os.path.join(data_dir, split_name)
        image_dir = os.path.join(split_dir, "images")
        text_dir = os.path.join(split_dir, "texts")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        # Ghi label.csv
        with open(os.path.join(split_dir, "label.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            writer.writerows(split_samples)

        # Copy ảnh và văn bản
        for id_, _ in split_samples:
            shutil.copy(os.path.join(data_dir, "images", f"{id_}.jpg"),
                        os.path.join(image_dir, f"{id_}.jpg"))
            shutil.copy(os.path.join(data_dir, "texts", f"{id_}.txt"),
                        os.path.join(text_dir, f"{id_}.txt"))

        print(f"[INFO] Chia {split_name}: {len(split_samples)} mẫu → {split_dir}")

# Ví dụ chạy
split_dataset("data", train_ratio=0.8, val_ratio=0.1)
