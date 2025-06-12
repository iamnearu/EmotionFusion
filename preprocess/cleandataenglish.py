import os
import re

def clean_text(text):
    text = re.sub(r'[@#]\w+', '', text)       # Bỏ @username, #hashtag
    text = re.sub(r'http\S+', '', text)       # Bỏ URL
    text = re.sub(r'[^\w\s]', '', text)       # Bỏ ký tự đặc biệt (ngoại trừ chữ và số)
    text = re.sub(r'\s+', ' ', text).strip()  # Chuẩn hóa khoảng trắng
    return text

# Đường dẫn đến thư mục chứa các file txt
input_dir = r"/data/texts/part4"

# Duyệt và xử lý từng file, ghi đè nội dung đã làm sạch
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        cleaned_text = clean_text(raw_text)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

print("✅ Đã làm sạch và ghi đè lên toàn bộ file trong thư mục.")