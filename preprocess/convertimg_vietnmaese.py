import os
from deep_translator import GoogleTranslator

# Đường dẫn tới thư mục chứa dữ liệu
data_folder = r"C:\Users\Iamnearu\Documents\MachineLearning\EmotionFusion\dataset\texts\part4"

# Duyệt qua tất cả file trong thư mục
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)

        # Đọc nội dung tiếng Anh
        with open(file_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        # Dịch sang tiếng Việt
        try:
            translated_text = GoogleTranslator(source='auto', target='vi').translate(original_text)

            # Ghi đè file cũ hoặc lưu file mới
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(translated_text)

            print(f"✅ Đã dịch: {filename}")
        except Exception as e:
            print(f"Lỗi khi dịch {filename}: {e}")