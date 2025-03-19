import json
import torch
import torchaudio
from torch.utils.data import Dataset
from feature_extractor import get_feature_extractor
import re

# Load từ vựng
with open("word2index.json", "r", encoding="utf-8") as f:
    word2index = json.load(f)

# Token đặc biệt
sos_token = word2index["<s>"]
eos_token = word2index["</s>"]
pad_token = word2index["<pad>"]

class SpeechDataset(Dataset):
    def __init__(self, json_path, feature_type="mel_spectrogram"):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = [
            {"audio_filepath": f"data/audio/{v['voice']}", "text": v["script"]}
            for v in raw_data.values() if "script" in v and "voice" in v
        ]

        self.feature_extractor = get_feature_extractor(feature_type=feature_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load file âm thanh
        waveform, sr = torchaudio.load(item["audio_filepath"])
        features = self.feature_extractor(waveform).squeeze(0).transpose(0, 1)  # [T, feature_dim]

        # Xử lý văn bản
        text = re.sub(r"[^\w\s]", " ", item["text"].lower()).strip()
        text = re.sub(r"\s+", " ", text)

        # Chuyển văn bản thành token
        encoded_text = [word2index.get(word, word2index["<unk>"]) for word in text.split()]
        tokens_bos = [sos_token] + encoded_text
        tokens_eos = encoded_text + [eos_token]

        return {
            "features": features,
            "tokens": torch.tensor(encoded_text, dtype=torch.long),
            "tokens_bos": torch.tensor(tokens_bos, dtype=torch.long),
            "tokens_eos": torch.tensor(tokens_eos, dtype=torch.long),
            "feat_length": features.shape[0],
            "target_length": len(encoded_text) + 2  # Thêm <s> và </s>
        }
