import torch
import yaml
import torchaudio
from dataset.dataset import SpeechRNNTDataset
from dataset.collate_fn import collate_fn
from models.transformer_transducer import TransformerTransducer
from torch.utils.data import DataLoader

# Load cấu hình
with open("/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Khởi tạo dataset
train_dataset = SpeechRNNTDataset(config["train_dataset"], feature_type=config["feature_extractor"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

dev_dataset = SpeechRNNTDataset(config["valid_dataset"], feature_type=config["feature_extractor"])
dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

# Khởi tạo mô hình
model = TransformerTransducer(input_dim=config["input_dim"], vocab_size=config["vocab_size"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Định nghĩa loss function và optimizer
criterion = torchaudio.transforms.RNNTLoss(blank=config["blank_idx"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# Vòng lặp huấn luyện
for epoch in range(config["epochs"]):
    model.train()
    for features, feature_lengths, labels, label_lengths in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features, feature_lengths, labels)
        loss = criterion(logits, labels, feature_lengths, label_lengths)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

    # Đánh giá trên tập dev
    model.eval()
    dev_loss = 0.0
    with torch.no_grad():
        for features, feature_lengths, labels, label_lengths in dev_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features, feature_lengths, labels)
            loss = criterion(logits, labels, feature_lengths, label_lengths)
            dev_loss += loss.item()
    dev_loss /= len(dev_loader)
    print(f"Validation Loss: {dev_loss:.4f}")

# Lưu lại mô hình
torch.save(model.state_dict(), "transformer_transducer.pth")
