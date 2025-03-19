import json
import torch
import yaml
import torchaudio
from dataset import SpeechDataset
from collate_fn import collate_fn
from transformer_transducer import TransformerTransducer
from torch.utils.data import DataLoader
from jiwer import wer, cer

# Load config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
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
    pred_embed_dim=config["pred_embed_dim"],
    pred_hidden_dim=config["pred_hidden_dim"],
    pred_layers=config["pred_layers"],
    joint_dim=config["joint_dim"],
    dropout=config["dropout"],
    blank_idx=config["blank_idx"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        logits = model(features, tokens_bos)
        loss = criterion(logits, tokens_eos, feat_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
