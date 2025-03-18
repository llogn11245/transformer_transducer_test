import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Tính toán trước ma trận vị trí
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Công thức tần số cho vị trí (sinusoids)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # vị trí chẵn dùng sin
        pe[:, 1::2] = torch.cos(position * div_term)  # vị trí lẻ dùng cos
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # lưu pe như một buffer không cần grad

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Cộng vector vị trí tương ứng vào embedding đầu vào
        x = x + self.pe[:, :seq_len, :]
        return x

# Bộ mã hóa Encoder sử dụng TransformerEncoder nhiều lớp
class AudioEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, n_heads, ffn_dim, dropout=0.1):
        """
        input_dim: số chiều của đầu vào (đặc trưng âm thanh, ví dụ n_mels)
        d_model: kích thước ẩn mô hình Transformer
        num_layers: số lớp TransformerEncoder
        n_heads: số head của Multi-head Attention
        ffn_dim: kích thước ẩn của Feedforward layer trong Transformer
        dropout: tỷ lệ dropout
        """
        super().__init__()
        # Linear để đưa đặc trưng âm thanh về kích thước d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional Encoding để thêm thông tin vị trí thời gian
        self.pos_enc = PositionalEncoding(d_model)

        # Khởi tạo các lớp TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features, lengths):
        """
        features: tensor âm thanh đã trích xuất đặc trưng, shape (batch, T, input_dim)
        lengths: độ dài (số frame) thực của mỗi mẫu trong batch, shape (batch,)
        """
        batch_size, max_len, _ = features.shape
        # Tạo mask để Transformer bỏ qua các frame đã padding (True tại vị trí được padding)
        # Mask shape: (batch, max_len)
        device = features.device
        mask = (torch.arange(max_len, device=device)[None, :] >= lengths[:, None].to(device))

        # Dựng đầu vào cho Transformer
        x = self.input_proj(features)             # (batch, T, d_model)
        x = self.pos_enc(x)                      # (batch, T, d_model) đã cộng pos encoding
        x = x.transpose(0, 1)                    # (T, batch, d_model) chuyển về shape phù hợp cho Transformer

        # Chạy qua TransformerEncoder
        enc_output = self.transformer_encoder(x, src_key_padding_mask=mask)  # output shape: (T, batch, d_model)
        enc_output = enc_output.transpose(0, 1)   # chuyển lại thành (batch, T, d_model)

        return enc_output  # trả về đặc trưng đã mã hóa và có cùng độ dài T (bao gồm padding frames)