from collections import Counter

def get_majority_label(text_labels):
    labels = [label.split(',')[0].strip() for label in text_labels]  # lấy phần 'text'
    counter = Counter(labels)
    most_common = counter.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    elif most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    else:
        return 'neutral'  # trường hợp hòa, bạn có thể chọn bỏ qua nếu muốn

input_file = r"C:\Users\Iamnearu\Documents\Học Máy\MVSA-multiple\MVSA\labelResultAll.txt"
output_file = r"C:\Users\Iamnearu\Documents\Học Máy\EmotionFusion\dataset\labels.csv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("ID,final_label\n")  # header
    for line in lines[1:]:  # bỏ header dòng đầu tiên
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            id = parts[0]
            majority = get_majority_label(parts[1:4])
            f_out.write(f"{id},{majority}\n")
