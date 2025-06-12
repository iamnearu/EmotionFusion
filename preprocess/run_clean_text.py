# preprocess/run_clean_text.py

import os
from clean_text import clean_text, load_stopwords

input_dir = r"C:\Users\Iamnearu\Documents\MachineLearning\EmotionFusion\data\texts"
output_dir = r"C:\Users\Iamnearu\Documents\MachineLearning\EmotionFusion\data\clean_texts"
stopword_file = r"C:\Users\Iamnearu\Documents\MachineLearning\EmotionFusion\preprocess\vietnamese-stopwords.txt"

os.makedirs(output_dir, exist_ok=True)
stopwords = load_stopwords(stopword_file)

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text, stopwords)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

print("✅ Đã xử lý và lưu kết quả vào:", output_dir)
