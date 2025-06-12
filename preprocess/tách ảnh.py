import os
import shutil

# Đường dẫn thư mục gốc
src_dir = r'/data\img_text'
# Đường dẫn mới để chứa ảnh và văn bản
image_dir = '../data/images'
text_dir = '../data/texts'

# Tạo thư mục mới nếu chưa có
os.makedirs(image_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

# Duyệt các file trong thư mục gốc
for filename in os.listdir(src_dir):
    filepath = os.path.join(src_dir, filename)

    # Nếu là ảnh
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.move(filepath, os.path.join(image_dir, filename))
    # Nếu là file văn bản
    elif filename.lower().endswith('.txt'):
        shutil.move(filepath, os.path.join(text_dir, filename))

print("✅ Đã tách xong ảnh và văn bản.")
