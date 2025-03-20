import os
import json
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset

# def create_vocab_from_data(train_json, vocab_path, use_subword=False):
#     """
#     Tạo file vocabulary từ dữ liệu huấn luyện nếu chưa tồn tại.
#     Với use_subword=True, huấn luyện mô hình SentencePiece để tạo subword vocab.
#     Với use_subword=False, tạo vocab gồm các ký tự duy nhất trong tập huấn luyện.
#     """
#     with open(train_json, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     # Thu thập tất cả transcript từ file train_json
#     texts = [entry["script"] for entry in data.values()]
#     if use_subword:
#         # Sử dụng thư viện SentencePiece để tạo vocab subword
#         import sentencepiece as spm

#         # Ghi toàn bộ transcript ra file tạm để huấn luyện SentencePiece
#         with open("all_text.txt", "w", encoding='utf-8') as fw:
#             for t in texts:
#                 fw.write(t + "\n")

#         # Huấn luyện SentencePiece (vocab_size có thể điều chỉnh tùy bài toán)
#         spm.SentencePieceTrainer.train(input='all_text.txt', model_prefix='spm', vocab_size=5000, character_coverage=1.0)

#         # Đọc vocab từ file spm.vocab do SentencePiece tạo ra
#         with open('spm.vocab', 'r', encoding='utf-8') as vf:
#             spm_vocab = [line.split('\t')[0] for line in vf]  # token nằm ở cột đầu

#         # Thêm các token đặc biệt vào đầu danh sách vocab
#         vocab_list = ['<pad>', '<sos>', '<eos>'] + spm_vocab
#     else:
#         # Tạo tập ký tự duy nhất từ toàn bộ transcript
#         all_text = "\n".join(texts)
#         chars = sorted(set(all_text))

#         # Thêm token đặc biệt (pad, sos, eos) vào đầu danh sách
#         vocab_list = ['<pad>', '<sos>', '<eos>'] + chars

#     # Ghi vocabulary ra file
#     with open(vocab_path, 'w', encoding='utf-8') as f:
#         for token in vocab_list:
#             f.write(token + "\n")

#     return vocab_list

# def create_vocab_from_data(train_json, vocab_path, use_subword=False):
#     """
#     Tạo file vocabulary từ dữ liệu huấn luyện nếu chưa tồn tại.
#     Với use_subword=True, huấn luyện mô hình SentencePiece để tạo subword vocab.
#     Với use_subword=False, tạo vocab gồm các ký tự duy nhất trong tập huấn luyện.
#     """
#     with open(train_json, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     # Thu thập tất cả transcript từ file train_json
#     texts = [entry["script"] for entry in data.values()]
    
#     if use_subword:
#         # Sử dụng thư viện SentencePiece để tạo vocab subword
#         import sentencepiece as spm
#         # Ghi toàn bộ transcript ra file tạm để huấn luyện SentencePiece
#         with open("all_text.txt", "w", encoding='utf-8') as fw:
#             for t in texts:
#                 fw.write(t + "\n")
#         # Huấn luyện SentencePiece (vocab_size có thể điều chỉnh tùy bài toán)
#         spm.SentencePieceTrainer.train(input='all_text.txt', model_prefix='spm', vocab_size=5000, character_coverage=1.0)
#         # Đọc vocab từ file spm.vocab do SentencePiece tạo ra
#         with open('spm.vocab', 'r', encoding='utf-8') as vf:
#             spm_vocab = [line.split('\t')[0] for line in vf]  # token nằm ở cột đầu
#         # Thêm các token đặc biệt vào đầu danh sách vocab
#         vocab_list = ['<pad>', '<sos>', '<eos>'] + spm_vocab
#     else:
#         # Tạo tập ký tự duy nhất từ toàn bộ transcript
#         all_text = "\n".join(texts)
#         chars = sorted(set(all_text))

#         # Đảm bảo rằng khoảng trắng (' ') được thêm vào vocab nếu chưa có
#         if ' ' not in chars:
#             chars.append(' ')
#         chars = sorted(chars)

#         # Thêm token đặc biệt (pad, sos, eos) vào đầu danh sách
#         vocab_list = ['<pad>', '<sos>', '<eos>'] + chars

#     # Ghi vocabulary ra file
#     with open(vocab_path, 'w', encoding='utf-8') as f:
#         for token in vocab_list:
#             f.write(token + "\n")
#     return vocab_list

def create_vocab_from_data(train_json, vocab_path, use_subword=False):
    """
    Tạo file vocabulary từ dữ liệu huấn luyện nếu chưa tồn tại.
    Với use_subword=True, huấn luyện mô hình SentencePiece để tạo subword vocab.
    Với use_subword=False, tạo vocab gồm các ký tự duy nhất trong tập huấn luyện.
    """
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Thu thập tất cả transcript từ file train_json
    texts = [entry["script"] for entry in data.values()]
    
    if use_subword:
        # Sử dụng thư viện SentencePiece để tạo vocab subword
        import sentencepiece as spm
        # Ghi toàn bộ transcript ra file tạm để huấn luyện SentencePiece
        with open("all_text.txt", "w", encoding='utf-8') as fw:
            for t in texts:
                fw.write(t + "\n")
        # Huấn luyện SentencePiece (vocab_size có thể điều chỉnh tùy bài toán)
        spm.SentencePieceTrainer.train(input='all_text.txt', model_prefix='spm', vocab_size=5000, character_coverage=1.0)
        # Đọc vocab từ file spm.vocab do SentencePiece tạo ra
        with open('spm.vocab', 'r', encoding='utf-8') as vf:
            spm_vocab = [line.split('\t')[0] for line in vf]  # token nằm ở cột đầu
        # Thêm các token đặc biệt vào đầu danh sách vocab
        vocab_list = ['<pad>', '<sos>', '<eos>', '<blank>'] + spm_vocab
    else:
        # Tạo tập ký tự duy nhất từ toàn bộ transcript
        all_text = "\n".join(texts)
        chars = sorted(set(all_text))

        # # Loại bỏ các ký tự không hợp lệ và đảm bảo thêm đúng khoảng trắng
        # chars = [ch for ch in chars if ch.strip() or ch == ' ']

        # # Đảm bảo rằng khoảng trắng (' ') được thêm vào vocab nếu chưa có
        # if ' ' not in chars:
        #     chars.append(' ')
        # chars = sorted(chars)

        # Thêm token đặc biệt (pad, sos, eos) vào đầu danh sách
        vocab_list = ['<pad>', '<sos>', '<eos>', '<blank>'] + chars

    # Tạo từ điển vocab: {token: index}
    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}

    # Ghi vocab_dict ra file JSON
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    
    print(f"Vocab đã được tạo và lưu tại {vocab_path}")
    return vocab_dict

class SpeechDataset(Dataset):
    """
    Dataset cho tác vụ ASR Transformer-Transducer.
    Mỗi phần tử gồm đặc trưng mel-spectrogram của âm thanh và chuỗi ký tự (đã mã hóa số) tương ứng.
    """
    def __init__(self, json_path, wav_folder, vocab, sample_rate=16000, use_subword=False):
        super(SpeechDataset, self).__init__()
        self.sample_rate = sample_rate
        self.use_subword = use_subword

        # Đọc file JSON để lấy danh sách audio và transcript
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Lưu danh sách (đường dẫn audio, transcript text)
        self.samples = []
        for entry in data.values():
            audio_file = os.path.join(wav_folder, entry["voice"])
            transcript = entry["script"]
            self.samples.append((audio_file, transcript))

        # Thiết lập vocabulary và từ điển ánh xạ token->index
        self.vocab = vocab
        self.vocab_dict = {token: idx for idx, token in enumerate(vocab)}

        # Nếu dùng subword, load mô hình SentencePiece để encode transcript
        if use_subword:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load('spm.model')  # mô hình SentencePiece đã được huấn luyện ở bước tạo vocab

        # Thiết lập bộ biến đổi mel-spectrogram
        self.n_mels = 80  # số mel filter-bank
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
        if self.use_subword:
            # Sử dụng mô hình SentencePiece để tách câu thành các token subword
            tokens = self.sp.encode_as_pieces(text.strip())
            indices = [self.vocab_dict.get(token, self.vocab_dict.get('<unk>', 0)) for token in tokens]
        else:
            # Với ký tự, bao gồm cả khoảng trắng như một ký tự
            indices = [self.vocab_dict[ch] for ch in text]

        # Thêm ký hiệu bắt đầu và kết thúc câu
        sos_id = self.vocab_dict.get('<sos>')
        eos_id = self.vocab_dict.get('<eos>')

        if sos_id is not None:
            indices = [sos_id] + indices
        if eos_id is not None:
            indices = indices + [eos_id]
        target_tensor = torch.LongTensor(indices)

        return mel_spec, target_tensor

# def collate_fn(batch):
#     """
#     Hàm collate để gom các mẫu vào một batch, padding cho âm thanh và transcript.
#     Trả về: 
#       - input_tensor: Tensor kích thước (B, n_mels, T_max) chứa mel-spectrogram đã padding 
#       - input_lengths: độ dài thật của từng input (số frame mel) 
#       - target_tensor: Tensor (B, U_max) chứa chỉ số ký tự đã padding 
#       - target_lengths: độ dài thật của từng chuỗi target (số token)
#     """
#     batch_size = len(batch)

#     # Sắp xếp batch theo độ dài audio giảm dần (tùy chọn, giúp hiệu quả tính toán)
#     batch.sort(key=lambda x: x[0].shape[1], reverse=True)
#     mels = [item[0] for item in batch]
#     targets = [item[1] for item in batch]

#     # Padding cho chuỗi âm thanh (mel-spectrogram)
#     n_mels = mels[0].shape[0]
#     max_len = max([m.shape[1] for m in mels])
#     input_tensor = torch.zeros(batch_size, n_mels, max_len)         # mặc định pad bằng 0
#     input_lengths = torch.zeros(batch_size, dtype=torch.long)

#     for i, m in enumerate(mels):
#         seq_len = m.shape[1]
#         input_tensor[i, :, :seq_len] = m        # chép dữ liệu vào phần đầu, phần dư vẫn là 0 (pad)
#         input_lengths[i] = seq_len

#     # Padding cho chuỗi target (chuỗi ký tự)
#     max_tgt_len = max([t.size(0) for t in targets])
#     target_tensor = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)  # pad bằng 0 (<pad>)
#     target_lengths = torch.zeros(batch_size, dtype=torch.long)

#     for i, t in enumerate(targets):
#         l = t.size(0)
#         target_tensor[i, :l] = t            # phần còn lại giữ ở 0 (pad)
#         target_lengths[i] = l
        
#     return input_tensor, input_lengths, target_tensor, target_lengths

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
