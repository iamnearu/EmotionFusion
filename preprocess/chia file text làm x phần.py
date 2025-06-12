import os
import shutil
from math import ceil

# Thư mục chứa các file .txt gốc
text_dir = r'/data/texts'

# Lấy danh sách tất cả các file .txt
all_txt_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
total_files = len(all_txt_files)

# Tổng số file là 19.600
num_parts = 4
chunk_size = ceil(total_files / num_parts)

print(f"Tổng số file: {total_files}, mỗi phần ~ {chunk_size} file")

# Chia và di chuyển
for i in range(num_parts):
    part_dir = os.path.join(text_dir, f'part{i + 1}')
    os.makedirs(part_dir, exist_ok=True)

    # Lấy file tương ứng cho từng phần
    part_files = all_txt_files[i * chunk_size:(i + 1) * chunk_size]

    for f in part_files:
        src = os.path.join(text_dir, f)
        dst = os.path.join(part_dir, f)
        shutil.move(src, dst)

print("✅ Đã chia xong file .txt thành 4 phần.")
