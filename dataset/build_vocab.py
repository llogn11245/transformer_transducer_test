import json
from tokenizer import Tokenizer

# Đường dẫn đến tập dữ liệu huấn luyện
TRAIN_DATASET_JSON = "data/train_metadata.json"  # Cập nhật đường dẫn thực tế nếu khác
VOCAB_FILE = "vocab.json"

def extract_text_samples(json_path):
    """
    Trích xuất danh sách văn bản từ tập dữ liệu huấn luyện.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    text_samples = []
    for key, value in raw_data.items():
        if "script" in value:
            text_samples.append(value["script"])
    return text_samples

if __name__ == "__main__":
    print("Đang xây dựng từ vựng...")
    text_samples = extract_text_samples(TRAIN_DATASET_JSON)
    tokenizer = Tokenizer(vocab_file=VOCAB_FILE)
    tokenizer.build_vocab(text_samples)
    print(f"Từ vựng đã được lưu vào {VOCAB_FILE}. Tổng số từ: {len(tokenizer.word2index)}")
