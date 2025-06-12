import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Mạng mã hóa ảnh sử dụng ResNet18 tiền huấn luyện để trích xuất đặc trưng hình ảnh.
    """
    def __init__(self, output_dim=256):
        super(ImageEncoder, self).__init__()

        # Load mô hình ResNet18 có trọng số pretrained
        resnet = models.resnet18(pretrained=True)

        # Bỏ lớp phân loại cuối cùng (fc), giữ lại phần backbone
        modules = list(resnet.children())[:-1]  # bỏ lớp fc
        self.backbone = nn.Sequential(*modules)

        # Fully connected layer để chuyển từ 512 -> output_dim
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): ảnh đầu vào [B, 3, H, W]
        Returns:
            Tensor: đặc trưng ảnh [B, output_dim]
        """
        with torch.no_grad():
            x = self.backbone(x)  # [B, 512, 1, 1]

        x = x.view(x.size(0), -1)  # [B, 512]
        x = self.fc(x)             # [B, output_dim]

        return x
