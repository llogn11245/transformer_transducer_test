import os
import json
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import torchaudio.transforms as T

class SpeechDataset(Dataset):
    """
    Dataset cho tác vụ ASR Transformer-Transducer.
    Mỗi phần tử gồm đặc trưng mel-spectrogram của âm thanh và chuỗi ký tự (đã mã hóa số) tương ứng.
    """
    def __init__(self, json_path, wav_folder, vocab_path, sample_rate=16000, transform_name="melspectrogram"):
        super(SpeechDataset, self).__init__()
        self.sample_rate = sample_rate

        # Đọc file JSON để lấy danh sách audio và transcript
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Lưu danh sách (đường dẫn audio, transcript text)
        self.samples = []
        for entry in data.values():
            audio_file = os.path.join(wav_folder, entry["voice"])
            transcript = entry["processed_script"]
            self.samples.append((audio_file, transcript))

        # Đọc vocab từ file word2index
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            self.vocab_dict = json.load(vf)
        
        # Lấy danh sách các từ (từ khóa của vocab_dict)
        self.vocab = list(self.vocab_dict.keys())

        # Lấy chỉ số của token <unk>
        self.unk_id = self.vocab_dict.get('<unk>', 0)

        self.n_mels = 80

        # Thiết lập bộ biến đổi âm thanh dựa trên tên
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]

        # Đọc âm thanh (có thể là .wav hoặc .mp3) và chuyển về sample_rate yêu cầu
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Nếu audio có nhiều kênh (stereo), trộn thành mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Tính toán mel-spectrogram từ waveform
        mel_spec = self.mel_transform(waveform)  # shape: (1, n_mels, time) nếu waveform là mono
        mel_spec = mel_spec.squeeze(0)           # bỏ dimension kênh, còn (n_mels, time)
        mel_spec = torch.log1p(mel_spec)         # lấy log của mel (log(1 + x)) để ổn định giá trị

        # Chuẩn hóa feature (mean-variance normalization theo từng utterance)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-10)

        # Mã hóa transcript thành chuỗi chỉ số
        words = text.strip().split(" ")
        
        indices = [self.vocab_dict.get(word, self.unk_id) for word in words]

        # Thêm ký hiệu bắt đầu và kết thúc câu
        sos_id = self.vocab_dict.get('<sos>', 1)
        eos_id = self.vocab_dict.get('<eos>', 2)

        if sos_id is not None:
            indices = [sos_id] + indices
        if eos_id is not None:
            indices = indices + [eos_id]
        
        target_tensor = torch.LongTensor(indices)

        return mel_spec, target_tensor

def collate_fn(batch):
    """
    Hàm collate để gom các mẫu vào một batch, padding cho âm thanh và transcript.
    Trả về: 
      - input_tensor: Tensor kích thước (B, n_mels, T_max) chứa mel-spectrogram đã padding 
      - input_lengths: độ dài thật của từng input (số frame mel) 
      - target_tensor: Tensor (B, U_max) chứa chỉ số ký tự đã padding 
      - target_lengths: độ dài thật của từng chuỗi target (số token)
    """
    batch_size = len(batch)
    # Sắp xếp batch theo độ dài audio giảm dần (giúp hiệu quả hơn trong RNN)
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    mels = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Padding cho âm thanh (mel-spectrogram)
    n_mels = mels[0].shape[0]
    max_len = max([mel.shape[1] for mel in mels])
    input_tensor = torch.zeros((batch_size, n_mels, max_len), dtype=torch.float32)
    input_lengths = torch.zeros((batch_size), dtype=torch.int32)
    for i, mel in enumerate(mels):
        seq_len = mel.shape[1]
        input_tensor[i, :, :seq_len] = mel
        input_lengths[i] = seq_len

    # Padding cho chuỗi nhãn (ký tự)
    max_tgt_len = max([len(t) for t in targets])
    target_tensor = torch.zeros((batch_size, max_tgt_len), dtype=torch.int32)  # Sửa kiểu int32
    target_lengths = torch.zeros((batch_size), dtype=torch.int32)

    for i, tgt in enumerate(targets):
        tgt_len = len(tgt)
        target_tensor[i, :tgt_len] = torch.tensor(tgt, dtype=torch.int32)
        target_lengths[i] = tgt_len
    
    return input_tensor, input_lengths, target_tensor, target_lengths