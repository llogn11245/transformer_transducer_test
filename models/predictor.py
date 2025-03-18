import torch
import torch.nn as nn

# Bộ tiên đoán Predictor sử dụng mô hình ngôn ngữ (LSTM nhiều lớp)
class Predictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        """
        vocab_size: kích thước vocab đầu ra (bao gồm ký hiệu blank)
        embed_dim: kích thước vector embedding cho mỗi token đầu ra
        hidden_dim: kích thước ẩn của LSTM
        num_layers: số lớp LSTM
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_seq):
        """
        input_seq: chuỗi ký tự đầu vào cho predictor (bao gồm ký hiệu <blank> ở đầu mỗi chuỗi)
                   shape: (batch, U+1) với U là độ dài chuỗi ký tự mục tiêu (không tính blank đầu)
        """
        # Embedding cho các token
        x = self.embed(input_seq)           # (batch, U+1, embed_dim)
        # Chạy qua LSTM
        # (Có thể sử dụng pack_padded_sequence để tối ưu nếu cần, nhưng ở đây đơn giản dùng trực tiếp)
        output, _ = self.lstm(x)            # output: (batch, U+1, hidden_dim)
        return output  # trả về chuỗi hidden states của predictor