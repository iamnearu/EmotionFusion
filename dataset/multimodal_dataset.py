import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultimodalEmotionDataset(Dataset):
    """
    Lớp Dataset để load các cặp dữ liệu đa phương thức: văn bản + ảnh + nhãn cảm xúc.
    Sử dụng trong quá trình huấn luyện hoặc đánh giá mô hình deep learning.
    """

    def __init__(self, data_dir, tokenizer, transform=None):
        """
        Khởi tạo Dataset.

        Args:
            data_dir (str): Thư mục chứa 'images/', 'texts/', và 'label.csv'
            tokenizer: Tokenizer dùng để mã hóa văn bản (ví dụ: PhoBERT tokenizer)
            transform: Các phép biến đổi ảnh (resize, normalize,...)
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform

        # Đọc file nhãn label.csv, lấy ID và nhãn cảm xúc tương ứng
        label_path = os.path.join(data_dir, "label.csv")
        self.samples = []
        with open(label_path, encoding='utf-8') as f:
            next(f)  # bỏ dòng header
            for line in f:
                id_, label = line.strip().split(',')
                self.samples.append((id_, label.strip().lower()))  # chuẩn hóa nhãn

        # Mapping nhãn cảm xúc từ tiếng Việt sang chỉ số số học
        self.label2id = {
            "tiêu cực": 0,
            "trung tính": 1,
            "tích cực": 2
        }

    def __len__(self):
        # Trả về số lượng mẫu trong dataset
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load một mẫu dữ liệu tại chỉ số `idx`, bao gồm:
        - ảnh đã transform
        - văn bản đã tokenize (input_ids, attention_mask)
        - nhãn cảm xúc (dưới dạng số)

        Returns:
            dict chứa các tensor cần thiết cho mô hình
        """
        id_, label = self.samples[idx]

        # --- Load văn bản ---
        text_path = os.path.join(self.data_dir, "clean_texts", f"{id_}.txt")
        with open(text_path, encoding='utf-8') as f:
            text = f.read().strip()

        # Mã hóa văn bản thành input_ids và attention_mask bằng tokenizer
        encoded = self.tokenizer(
            text,
            padding="max_length",  # đệm độ dài cố định
            truncation=True,  # cắt bớt nếu quá dài
            max_length=128,  # độ dài tối đa
            return_tensors="pt"  # trả về tensor PyTorch
        )

        # --- Load ảnh ---
        image_path = os.path.join(self.data_dir, "images", f"{id_}.jpg")
        image = Image.open(image_path).convert("RGB")  # chuyển sang RGB
        if self.transform:
            image = self.transform(image)  # áp dụng resize, normalize,...

        # --- Trả về đầu ra ---
        return {
            "input_ids": encoded["input_ids"].squeeze(0),  # [seq_len]
            "attention_mask": encoded["attention_mask"].squeeze(0),  # [seq_len]
            "image": image,  # [3, H, W]
            "label": torch.tensor(self.label2id[label])  # int64
        }
