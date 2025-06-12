# preprocess/clean_text.py

from underthesea import text_normalize, word_tokenize

def load_stopwords(stopword_file):
    """Load danh sách stopword từ file .txt (mỗi dòng 1 từ)."""
    with open(stopword_file, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def clean_text(text, stopwords=None):
    """
    Tiền xử lý văn bản:
        - Chuẩn hóa Unicode, chuyển về chữ thường
        - Tách từ bằng word_tokenize
        - Loại bỏ stopword và token không phải chữ
    """
    normalized = text_normalize(text).lower()
    tokens = word_tokenize(normalized, format="text").split()

    if stopwords:
        tokens = [tok for tok in tokens if tok.isalpha() and tok not in stopwords]
    else:
        tokens = [tok for tok in tokens if tok.isalpha()]

    return ' '.join(tokens)
