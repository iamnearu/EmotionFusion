import pandas as pd

# Đọc file gốc
df = pd.read_csv(r"/data\labels.csv")

# Mapping tiếng Anh -> tiếng Việt
label_map = {
    "positive": "tích cực",
    "neutral": "trung tính",
    "negative": "tiêu cực"
}

# Áp dụng mapping và thay thế trực tiếp cột final_label
df["final_label"] = df["final_label"].map(label_map)

# Ghi ra file mới
df.to_csv(r"C:\Users\Iamnearu\Documents\MachineLearning\EmotionFusion\dataset\lablel.csv", index=False, encoding="utf-8-sig")

print("✅ Hoàn tất: final_labels_vi.csv chỉ chứa nhãn tiếng Việt")
