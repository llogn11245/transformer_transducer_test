import torch
import torch.nn as nn

class JointNetwork(nn.Module):
    def __init__(self, enc_dim, pred_dim, joint_dim, vocab_size):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.joint_fc = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc_out, pred_out):
        # Chiếu encoder và predictor output về cùng độ dài joint_dim
        enc_proj = self.enc_proj(enc_out)   # (batch, T, joint_dim)
        pred_proj = self.pred_proj(pred_out)  # (batch, U+1, joint_dim)

        # Thêm chiều để broadcast cộng hai tensor
        enc_proj = enc_proj.unsqueeze(2)  # (batch, T, 1, joint_dim)
        pred_proj = pred_proj.unsqueeze(1)  # (batch, 1, U+1, joint_dim)

        # Tránh lỗi shape mismatch bằng cách kiểm tra kích thước
        U = pred_out.shape[1] - 1  # U là số ký tự target
        T = enc_out.shape[1]  # T là số frame đặc trưng

        # Cắt hoặc pad để đảm bảo kích thước tương thích
        if logits.shape[2] != U + 1:
            logits = logits[:, :, :U+1, :]


        # Cộng encoder và predictor (broadcast theo T và U+1), áp dụng kích hoạt tanh
        joint = torch.tanh(enc_proj + pred_proj)  # (batch, T, U+1, joint_dim)

        # Tính logits cho mỗi tổ hợp thời gian/nhãn
        logits = self.joint_fc(joint)  # (batch, T, U+1, vocab_size)
        
        return logits
