import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    """
    Mô hình phân loại cảm xúc dựa trên đặc trưng văn bản và hình ảnh.

    Cấu trúc tham khảo từ mô hình MulDIC:
    - Text và image được encode riêng, sau đó kết hợp bằng phép nhân phần tử (element-wise multiplication).
    - Đặc trưng hợp nhất được đưa qua 2 lớp fully-connected để phân loại.
    """

    def __init__(self, feature_dim=256, num_classes=3):
        """
        Args:
            feature_dim (int): Kích thước vector đặc trưng của mỗi modal (text/image).
            num_classes (int): Số lớp đầu ra (vd: 3 cảm xúc: tiêu cực, trung tính, tích cực).
        """
        super(FusionClassifier, self).__init__()

        # Hàm hợp nhất đặc trưng text và image bằng phép nhân phần tử (element-wise multiplication)
        self.fusion = lambda x_text, x_image: x_text * x_image  # [batch_size, feature_dim]

        # Mạng phân loại: gồm 2 lớp fully connected
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),  # giảm chiều đặc trưng xuống 128
            nn.ReLU(),                    # kích hoạt phi tuyến
            nn.Dropout(0.3),              # tránh overfitting
            nn.Linear(128, num_classes)   # đầu ra logits cho mỗi lớp cảm xúc
        )

    def forward(self, text_features, image_features):
        """
        Args:
            text_features (Tensor): Đặc trưng từ encoder văn bản, shape: [batch_size, feature_dim]
            image_features (Tensor): Đặc trưng từ encoder ảnh, shape: [batch_size, feature_dim]

        Returns:
            logits (Tensor): Đầu ra chưa softmax, shape: [batch_size, num_classes]
        """
        # Kết hợp đặc trưng từ text và image
        fused = self.fusion(text_features, image_features)  # [batch_size, feature_dim]

        # Đưa vào mạng phân loại để dự đoán logits
        logits = self.classifier(fused)  # [batch_size, num_classes]

        return logits
