# train.py
# Mã huấn luyện mô hình đa modal cảm xúc (image + text)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AdamWeightDecay, get_scheduler

from dataset.multimodal_dataset import MultimodalEmotionDataset
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_classifier import FusionClassifier

# ==== 1. Cài đặt ==== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
DATA_DIR = "data/train"

# ==== 2. Chuẩn bị Dataloader ==== #
print("\n[INFO] Loading dataset...")

# Tokenizer PhoBERT cho xử lý văn bản
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Tiền xử lý ảnh:
# - Resize tất cả ảnh về kích thước 258x258 để đảm bảo đồng nhất
# - Chuyển ảnh thành tensor
# - Chuẩn hóa các giá trị pixel về khoảng (-1, 1) bằng Normalize
transform = transforms.Compose([
    transforms.Resize((258, 258)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Tạo Dataset và DataLoader
dataset = MultimodalEmotionDataset(
    data_dir=DATA_DIR,
    tokenizer=tokenizer,
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== 3. Khởi tạo mô hình ==== #
print("[INFO] Building model...")
text_encoder = TextEncoder().to(DEVICE)  # Mã hóa đặc trưng văn bản
image_encoder = ImageEncoder().to(DEVICE)  # Mã hóa đặc trưng hình ảnh
fusion_model = FusionClassifier().to(DEVICE)  # Kết hợp và phân loại cảm xúc

# ==== 4. Tối ưu hóa và Loss ==== #
# Gộp tham số từ 3 mô hình để tối ưu hóa cùng nhau
params = list(text_encoder.parameters()) + \
         list(image_encoder.parameters()) + \
         list(fusion_model.parameters())

optimizer = AdamWeightDecay(params, lr=LR)
loss_fn = nn.CrossEntropyLoss()  # Phân loại nhiều lớp
scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(dataloader)
)

# ==== 5. Huấn luyện ==== #
print("[INFO] Start training...")
fusion_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # Forward qua text encoder và image encoder để lấy đặc trưng
        text_feat = text_encoder(input_ids, attention_mask)   # [B, 256]
        image_feat = image_encoder(images)                    # [B, 256]

        # Hợp nhất đặc trưng và phân loại cảm xúc
        logits = fusion_model(text_feat, image_feat)          # [B, 3]

        # Tính loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Tính độ chính xác
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")

# ==== 6. Lưu model ==== #
print("[INFO] Saving model...")
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    'text_encoder': text_encoder.state_dict(),
    'image_encoder': image_encoder.state_dict(),
    'fusion_model': fusion_model.state_dict()
}, "checkpoints/multimodal_model.pt")

print("[INFO] Training completed.")
