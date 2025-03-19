import json
import torch
import yaml
import torchaudio
from dataset.dataset import SpeechDataset
from dataset.collate_fn import collate_fn
from models.transformer_transducer import TransformerTransducer
from torch.utils.data import DataLoader
from jiwer import wer, cer

# Load config.yaml
with open("/content/transformer_transducer_test/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load số lượng từ vựng từ vocab
with open(config["vocab_path"], "r", encoding="utf-8") as f:
    word2index = json.load(f)
index2word = {v: k for k, v in word2index.items()}  
config["vocab_size"] = len(word2index)

# Khởi tạo dataset
train_dataset = SpeechDataset(config["train_dataset"], feature_type=config["feature_extractor"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

valid_dataset = SpeechDataset(config["valid_dataset"], feature_type=config["feature_extractor"])
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

# Khởi tạo mô hình
model = TransformerTransducer(
    input_dim=config["input_dim"], 
    vocab_size=config["vocab_size"],
    d_model=config["d_model"], 
    n_heads=config["n_heads"],
    ff_dim=config["ff_dim"], 
    enc_layers=config["enc_layers"],
    pred_embed_dim=config["pred_embed_dim"],  # Sửa lại đúng tên tham số
    pred_hidden_dim=config["pred_hidden_dim"],
    pred_layers=config["pred_layers"],
    joint_dim=config["joint_dim"],
    dropout=config["dropout"],
    blank_idx=config["blank_idx"]
)

device = torch.device("cpu")
model.to(device)

criterion = torchaudio.transforms.RNNTLoss(blank=config["blank_idx"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

def decode_prediction(pred):
    return " ".join([index2word[idx] for idx in pred if idx in index2word and idx != config["blank_idx"]])

for epoch in range(config["epochs"]):
    model.train()
    total_loss, total_wer, total_cer = 0, 0, 0

    for batch in train_loader:
        features, tokens_bos, tokens_eos, feat_lengths, target_lengths = (
            batch["features"].to(device),
            batch["tokens_bos"].to(device),
            batch["tokens_eos"].to(device),
            batch["feat_length"].to(device),
            batch["target_length"].to(device),
        )
        
        optimizer.zero_grad()
        logits = model(features, feat_lengths, tokens_eos)  # Truyền tokens_eos (target_sequence) vào đây
        
        target_lengths = torch.minimum(target_lengths, torch.tensor(tokens_eos.shape[1], device=target_lengths.device))
        feat_lengths = torch.minimum(feat_lengths, torch.tensor(logits.shape[1], device=feat_lengths.device))

        # Kiểm tra kích thước của logits
        print("Logits shape:", logits.shape)
        print("Tokens EOS shape:", tokens_eos.shape)
        print("Feat lengths:", feat_lengths)
        print("Target lengths:", target_lengths)

        
        # Chuyển đổi tokens_eos và feat_lengths sang kiểu int32
        tokens_eos = tokens_eos.to(torch.int32)
        feat_lengths = feat_lengths.to(torch.int32)
        target_lengths = target_lengths.to(torch.int32)
        
        # Tính loss
        loss = criterion(logits, tokens_eos, feat_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
