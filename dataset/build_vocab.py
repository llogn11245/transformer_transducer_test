import json
from collections import Counter

# Định nghĩa đường dẫn tệp dữ liệu đầu vào và đầu ra
DATASET_PATH = "/content/train.json"  # Cập nhật đường dẫn thực tế của bạn
VOCAB_TXT_PATH = "vocab.txt"
WORD2INDEX_PATH = "word2index.json"
INDEX2WORD_PATH = "index2word.json"

# Đọc dữ liệu
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [value["script"] for value in data.values() if "script" in value]

# Xây dựng từ vựng
def build_vocab(text_samples, min_freq=1):
    counter = Counter()
    for sentence in text_samples:
        counter.update(sentence.split())  # Token hóa theo từ (nếu cần có thể dùng tokenizer khác)

    # Thêm các token đặc biệt
    vocab = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab += [word for word, freq in counter.items() if freq >= min_freq]

    # Tạo word2index và index2word
    word2index = {word: idx for idx, word in enumerate(vocab)}
    index2word = {idx: word for word, idx in word2index.items()}

    return vocab, word2index, index2word

# Lưu vocab ra tệp
def save_vocab(vocab, word2index, index2word):
    with open(VOCAB_TXT_PATH, "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

    with open(WORD2INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(word2index, f, ensure_ascii=False, indent=4)

    with open(INDEX2WORD_PATH, "w", encoding="utf-8") as f:
        json.dump(index2word, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Đang xây dựng từ vựng...")
    text_samples = load_dataset(DATASET_PATH)
    vocab, word2index, index2word = build_vocab(text_samples)
    save_vocab(vocab, word2index, index2word)
    print(f"Từ vựng đã được tạo. Tổng số từ: {len(vocab)}")
