import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Bộ mã hóa văn bản sử dụng PhoBERT để trích xuất đặc trưng văn bản.
    """
    def __init__(self, model_name='vinai/phobert-base', output_dim=256):
        super(TextEncoder, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.phobert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # Trích xuất đầu ra của PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # lấy [CLS] token

        # Ánh xạ sang vector đặc trưng đầu ra
        x = self.dropout(pooled_output)
        x = self.linear(x)
        x = self.relu(x)
        return x  # kích thước [batch_size, output_dim]
