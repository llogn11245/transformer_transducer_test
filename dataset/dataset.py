import json
import torch
import torchaudio
from torch.utils.data import Dataset
from dataset.feature_extractor import get_feature_extractor
from tokenizer import Tokenizer

class SpeechRNNTDataset(Dataset):
    def __init__(self, json_path, feature_type="mel_spectrogram", vocab_file="vocab.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = []
        for key, value in raw_data.items():
            if "script" in value and "voice" in value:
                self.data.append({
                    "audio_filepath": f"data/audio/{value['voice']}",
                    "text": value["script"]
                })

        self.feature_extractor = get_feature_extractor(feature_type=feature_type)
        self.tokenizer = Tokenizer(vocab_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform, sr = torchaudio.load(item["audio_filepath"])
        features = self.feature_extractor(waveform).squeeze(0).transpose(0, 1)  # [T, feature_dim]
        token_indices = torch.tensor(self.tokenizer.encode(item["text"]))
        return features, token_indices
