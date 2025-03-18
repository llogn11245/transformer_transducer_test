import torch
import torch.nn as nn

# Mạng kết hợp Joint Network để hợp nhất encoder output và predictor output
class JointNetwork(nn.Module):
    def __init__(self, enc_dim, pred_dim, joint_dim, vocab_size):
        """
        enc_dim: kích thước ẩn của encoder output (d_model)
        pred_dim: kích thước ẩn của predictor output (hidden_dim của LSTM)
        joint_dim: kích thước ẩn của tầng joint (thường bằng hoặc nhỏ hơn enc_dim/pred_dim)
        vocab_size: kích thước từ vựng (số nhãn đầu ra, bao gồm blank)
        """
        super().__init__()

        # Linear để đưa enc_output và pred_output về cùng kích thước joint_dim
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)

        # Linear cuối để dự đoán phân phối xác suất cho vocab (bao gồm blank)
        self.joint_fc = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc_out, pred_out):
        """
        enc_out: tensor output từ encoder, shape (batch, T, enc_dim)
        pred_out: tensor output từ predictor, shape (batch, U+1, pred_dim)
        Trả về: logits (chưa qua softmax) với shape (batch, T, U+1, vocab_size)
        """
        # Chiếu encoder và predictor output về cùng độ dài joint_dim
        enc_proj = self.enc_proj(enc_out)   # (batch, T, joint_dim)
        pred_proj = self.pred_proj(pred_out)  # (batch, U+1, joint_dim)

        # Thêm chiều để broadcast cộng hai tensor
        # enc_proj_unsqueeze: (batch, T, 1, joint_dim)
        # pred_proj_unsqueeze: (batch, 1, U+1, joint_dim)
        enc_proj = enc_proj.unsqueeze(2)
        pred_proj = pred_proj.unsqueeze(1)

        # Cộng encoder và predictor (broadcast theo T và U+1), áp dụng kích hoạt tanh
        joint = torch.tanh(enc_proj + pred_proj)  # (batch, T, U+1, joint_dim)

        # Tính logits cho mỗi tổ hợp thời gian/nhãn
        logits = self.joint_fc(joint)  # (batch, T, U+1, vocab_size)
        
        return logits