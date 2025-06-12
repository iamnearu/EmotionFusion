EmotionFusion: Mô hình Học Sâu Đa phương thức cho Phân loại Cảm xúc
Dự án này tập trung vào việc xây dựng một mô hình học sâu đa phương thức để phân loại cảm xúc dựa trên cả dữ liệu văn bản và hình ảnh.

I. Quy trình Phát triển
Dự án được chia thành các giai đoạn chính như sau:

Giai đoạn 1: Tiền xử lý Dữ liệu
1.1. Tiền xử lý Dữ liệu Tiếng Anh
Bước 1: Làm sạch dữ liệu tiếng Anh.
Bước 2: Chạy script do "tiến bùi" cung cấp.
Bước 3: Chuyển đổi dữ liệu sang tiếng Việt (sử dụng file do "cương" cung cấp).
1.2. Tiền xử lý Dữ liệu Tiếng Việt
Văn bản (Text):

Chuẩn hóa chữ viết: Chuyển toàn bộ văn bản về chữ thường.
Tokenization: Sử dụng thư viện underthesea để tách từ (tokenize).
Xử lý từ dừng (Stopwords): Loại bỏ các từ dừng phổ biến trong tiếng Việt.
Hình ảnh (Image):

Resize ảnh: Thay đổi kích thước tất cả các ảnh về 258x258 pixels.
Mục tiêu: Đảm bảo mọi ảnh đầu vào đều có kích thước đồng nhất cho mô hình.
Chuẩn hóa pixel: Chuẩn hóa giá trị pixel về khoảng [-1, 1].
Mục tiêu: Giúp mô hình học ổn định và nhanh hơn bằng cách chuẩn hóa phân phối dữ liệu đầu vào.
Giai đoạn 2: Mô hình Học sâu Đa phương thức (Multimodal Deep Learning Model)
2.1. Kiến trúc Tổng quan:
Dự án sử dụng kiến trúc học sâu đa phương thức, kết hợp các encoder cho từng loại dữ liệu và một lớp hợp nhất (fusion layer) để kết nối chúng.

Text Encoder:
Lựa chọn: Sử dụng BERT tiếng Việt (ví dụ: PhoBERT, viBERT), hoặc kiến trúc embedding + LSTM/CNN.
Image Encoder:
Lựa chọn: Sử dụng mạng CNN (ví dụ: ResNet18, ResNet50, EfficientNet...).
Fusion Layer:
Phương pháp: Concatenation (nối) hoặc Element-wise multiplication (nhân từng phần tử).
Classifier:
Cấu trúc: Các lớp Dense (fully connected) dẫn đến lớp Softmax để phân loại nhãn cảm xúc.
2.2. Chi tiết Triển khai Kiến trúc:
Bước	Mô tả
1	Sử dụng PhoBERT (hoặc BERT VN) làm Text Encoder.
2	Sử dụng ResNet hoặc ViT làm Image Encoder.
3	Trích xuất embedding (vector đặc trưng) từ mỗi modal (văn bản và hình ảnh).
4	Thực hiện Fusion (hợp nhất) bằng element-wise multiplication (hoặc thử concatenation + MLP).
5	Sử dụng các lớp Dense + Softmax để phân loại nhãn cảm xúc.
6	Huấn luyện và đánh giá hiệu quả của mô hình (dựa trên F1 score, v.v.).

Export to Sheets
Giai đoạn 3: Huấn luyện Mô hình
Chia dữ liệu: Dữ liệu sẽ được chia thành các tập Train, Validation và Test.
Hàm mất mát (Loss Function): Sử dụng CrossEntropyLoss.
Bộ tối ưu hóa (Optimizer): Sử dụng Adam hoặc AdamW.
Scheduler: Sử dụng scheduler nếu cần thiết để điều chỉnh tốc độ học.
Theo dõi: Theo dõi các chỉ số loss, accuracy, và F1 score trong quá trình huấn luyện.
Giai đoạn 4: Đánh giá & Xuất kết quả
Chỉ số đánh giá: Tính toán Precision, Recall và F1-Score.
Ma trận nhầm lẫn: Vẽ Confusion Matrix để trực quan hóa hiệu suất phân loại.
Kiểm tra mẫu cụ thể: Thực hiện kiểm tra trên vài mẫu dữ liệu cụ thể để đánh giá chất lượng dự đoán.
Giai đoạn 5: Triển khai (Tùy chọn)
API: Đưa mô hình lên Flask / FastAPI.
Giao diện Web: Xây dựng một trang web cho phép người dùng tải ảnh và nhập văn bản, sau đó trả về nhãn cảm xúc dự đoán.