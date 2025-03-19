import torch
import torch.nn as nn

# Bộ tiên đoán Predictor sử dụng mô hình ngôn ngữ (LSTM nhiều lớp)
class Predictor(nn.Module):
    def __init__(self, vocab_size, pred_embed_dim, pred_hidden_dim, num_layers, blank_idx):
        """
        vocab_size: kích thước vocab đầu ra (bao gồm ký hiệu blank)
        pred_embed_dim: kích thước vector embedding cho mỗi token đầu ra
        pred_hidden_dim: kích thước ẩn của LSTM
        num_layers: số lớp LSTM
        blank_idx: chỉ mục của token blank trong vocab
        """
        super().__init__()
        self.blank_idx = blank_idx  # Lưu giá trị blank_idx
        self.embed = nn.Embedding(vocab_size, pred_embed_dim)
        self.lstm = nn.LSTM(pred_embed_dim, pred_hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = pred_hidden_dim

    def forward(self, input_seq):
        """
        input_seq: chuỗi ký tự đầu vào cho predictor (bao gồm ký hiệu <blank> ở đầu mỗi chuỗi)
                   shape: (batch, U+1) với U là độ dài chuỗi ký tự mục tiêu (không tính blank đầu)
        """
        # Mask bỏ qua blank_idx trong embedding
        input_seq = torch.where(input_seq == self.blank_idx, torch.tensor(0, device=input_seq.device), input_seq)

        # Embedding cho các token
        x = self.embed(input_seq)           # (batch, U+1, pred_embed_dim)

        # Chạy qua LSTM
        output, _ = self.lstm(x)            # output: (batch, U+1, pred_hidden_dim)
        
        return output  # Trả về chuỗi hidden states của predictor
