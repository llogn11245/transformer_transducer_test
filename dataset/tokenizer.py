import json
import os
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_file='vocab.json', min_freq=1):
        self.vocab_file = vocab_file
        self.min_freq = min_freq
        self.word2index = {}
        self.index2word = []
        if os.path.exists(vocab_file):
            self.load_vocab()

    def build_vocab(self, text_data):
        """
        Xây dựng vocab từ danh sách văn bản.
        """
        counter = Counter()
        for text in text_data:
            counter.update(text)

        vocab = ['<pad>', '<unk>', '<blank>']  # Các token đặc biệt
        vocab += [word for word, freq in counter.items() if freq >= self.min_freq]

        self.word2index = {word: idx for idx, word in enumerate(vocab)}
        self.index2word = vocab

        self.save_vocab()

    def save_vocab(self):
        """Lưu vocab vào file JSON."""
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump({'word2index': self.word2index, 'index2word': self.index2word}, f, ensure_ascii=False, indent=4)

    def load_vocab(self):
        """Tải vocab từ file JSON."""
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word2index = data['word2index']
            self.index2word = data['index2word']

    def encode(self, text):
        """
        Chuyển văn bản thành danh sách index.
        """
        return [self.word2index.get(char, self.word2index['<unk>']) for char in text]

    def decode(self, indices):
        """
        Chuyển danh sách index thành văn bản.
        """
        return ''.join([self.index2word[idx] for idx in indices if idx < len(self.index2word)])

# Nếu chạy file này trực tiếp, nó sẽ tạo vocab từ tập dữ liệu mẫu
if __name__ == "__main__":
    text_samples = ["xin chào", "học máy", "xử lý ngôn ngữ tự nhiên"]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(text_samples)
    print("Vocab size:", len(tokenizer.word2index))
