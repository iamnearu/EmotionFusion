
# 🤖 Multimodal Emotion Classification (Text + Image)

Dự án này xây dựng một hệ thống **phân tích cảm xúc đa phương thức** từ văn bản và hình ảnh sử dụng mô hình deep learning gồm ba thành phần chính: `TextEncoder`, `ImageEncoder`, và `FusionClassifier`.

---

## 📌 Mục tiêu

Phân loại cảm xúc đầu ra thành 3 lớp:
- **Tích cực**
- **Trung tính**
- **Tiêu cực**

Dữ liệu đầu vào bao gồm:
- 📝 **Văn bản**: mô tả, bình luận, trạng thái, v.v.
- 🖼️ **Hình ảnh**: ảnh đại diện, ảnh tình huống kèm theo

---

## 🏗️ Kiến trúc mô hình

```
          Text (.txt)          Image (.jpg)
               │                     │
               ▼                     ▼
         ┌──────────┐         ┌─────────────┐
         │Tokenizer │         │ Image Resize│
         └────┬─────┘         └────┬────────┘
              ▼                       ▼
     ┌────────────────┐      ┌─────────────────┐
     │  TextEncoder    │      │  ImageEncoder   │
     │ (PhoBERT-base)  │      │ (ResNet18, etc) │
     └────┬────────────┘      └────┬────────────┘
          ▼                       ▼
      Text vector            Image vector
       (dim=256)              (dim=256)
               \             /
                ▼           ▼
              ┌──────────────┐
              │ FusionClassifier │
              └──────┬───────────┘
                     ▼
            🔮 Predict Emotion (0/1/2)
```

---

## 🧩 Các thành phần chính

### 1. `TextEncoder` (PhoBERT-based)
- Sử dụng mô hình `vinai/phobert-base`
- Pooling (CLS/Mean/Max)
- Linear projection → 256 chiều
- Kèm dropout và GELU

### 2. `ImageEncoder`
- Backbone CNN: ResNet18 / ResNet50 / EfficientNet
- Tùy chọn `unfreeze_blocks` để fine-tune
- Output vector 256 chiều

### 3. `FusionClassifier`
- Kết hợp đặc trưng text và image qua attention và gated fusion
- Dự đoán cảm xúc 3 lớp

---

## 🧪 Huấn luyện & Đánh giá

- Optimizer: `AdamW`
- Learning rates: `2e-5` (encoder), `1e-4` (fusion)
- Loss: `CrossEntropyLoss` (có trọng số nhãn)
- Scheduler: `CosineAnnealingLR`
- Chỉ số theo dõi chính: `Weighted F1-score`
- Cơ chế `EarlyStopping` nếu không cải thiện sau `patience=8`

---

## 💾 Lưu mô hình

Mô hình tốt nhất được lưu tự động:
```python
torch.save({
  'text_encoder_state_dict': ...,
  'image_encoder_state_dict': ...,
  'fusion_model_state_dict': ...,
  ...
}, "checkpoints/best_multimodal_model.pt")
```

---

## ▶️ Demo (Inference)

```python
# Load mô hình và weights
# Chuẩn hóa văn bản + ảnh
# Truyền qua TextEncoder & ImageEncoder
# → FusionClassifier → Dự đoán cảm xúc
```

---

## 📁 Cấu trúc thư mục

```
project_root/
├── data/
│   ├── train/
│   │   ├── texts/*.txt
│   │   └── images/*.jpg
│   └── val/
├── checkpoints/
│   └── best_multimodal_model.pt
├── models/
│   ├── text_encoder.py
│   ├── image_encoder.py
│   └── fusion_classifier.py
├── dataset.py
├── train.py
├── evaluate.py
└── README.md
```

---

## ⚙️ Yêu cầu hệ thống

- Python ≥ 3.8
- PyTorch ≥ 1.13
- Transformers (HuggingFace)
- torchvision, pandas, Pillow, scikit-learn

---

## 📌 Tác giả

- 🧑 Nguyễn Huy Cương – Đại học Công nghiệp Hà Nội
- 📚 Đề tài nghiên cứu cá nhân – Đa phương thức và mô hình hóa cảm xúc
