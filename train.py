import os
import yaml
import json
import torchaudio
import datetime
from torch import nn
from model_builder import build_transformer_transducer
import torch
import speechbrain as sb 
from speechbrain.lobes.features import Fbank, MFCC
from speechbrain.dataio.dataloader import make_dataloader
from torch.utils.data import DataLoader
from speechbrain.dataio.batch import PaddedBatch

# Sử dụng hàm loss cho RNNT (cần cài đặt warp-transducer hoặc dùng torchaudio)
try:
    from torchaudio.functional import rnnt_loss  # warp-transducer loss
except ImportError:
    import torchaudio
    rnnt_criterion = lambda log_probs, targets, in_len, tgt_len: torchaudio.functional.rnnt_loss(
        log_probs, targets, in_len, tgt_len, blank=0, reduction='mean'
    )

word2index = {}
index2word = {}
def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        word2index = dict()
        index2word = dict()
    
        for index, char in enumerate(labels):
            word2index[char] = index
            index2word[index] = char
            
        return word2index, index2word

word2index, index2word = load_label_json(r'C:\paper\Transformer-Transducer-main\linhtinh\LSVSC_100_vocab.json') # Vocab file
print(len(word2index))
sos_token, eos_token, pad_token = word2index['<s>'], word2index['</s>'], word2index['_'] # <s>, </s>, _

@sb.utils.data_pipeline.takes("wav_file", "transcript")
@sb.utils.data_pipeline.provides("features", "encoded_text", "tokens_bos", "tokens_eos", "feat_length", "target_length")
def parse_characters(path, text):
    sig  = sb.dataio.dataio.read_audio(f"{path}")
    audio_feature = Fbank(sample_rate=16000, n_fft=512, n_mels=80)
    # features = hparams['compute_features'](sig.unsqueeze(0)).squeeze(0) # Đọc MFCC
    features = audio_feature(sig.unsqueeze(0)).squeeze(0)
    yield features
    
    encoded_text = [word2index.get(x, word2index["unk"]) for x in text.split(' ')]
    yield torch.LongTensor(encoded_text)
    yield torch.LongTensor([sos_token] + encoded_text)
    yield torch.LongTensor(encoded_text + [eos_token])

    yield features.size(0)

    yield len(encoded_text)+2

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_dataset(path, shuffle):
    data = load_json(path)

    if shuffle:
        keys = list(data.keys())
    else:
        dataset = sb.dataio.dataset.DynamicItemDataset(data)

    dataset.add_dynamic_item(parse_characters)
    dataset.set_output_keys(['id','features', 'encoded_text', 'tokens_bos', 'tokens_eos', 'feat_length', 'target_length'])
    return dataset

if __name__ == "__main__":
    # Đọc file cấu hình
    with open(r"C:\paper\Transformer-Transducer-main\config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    sample_rate = config.get("sample_rate", 16000)
    use_subword = config.get("use_subword", False)
    vocab_path = config["vocab_path"]
    train_json = config["train_json"]
    dev_json = config["dev_json"]
    test_json = config["test_json"]
    wav_folder = config["wav_folder"]

    train_dataset = get_dataset(train_json, False)
    dev_dataset = get_dataset(dev_json, False)
    test_dataset = get_dataset(test_json, False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=PaddedBatch
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1,
        collate_fn=PaddedBatch
    )
    vocab_size = len(word2index)

    # Thiết lập thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình Transformer-Transducer
    model = build_transformer_transducer(
        device=device, num_vocabs=vocab_size, input_size=80
    )
    model.to(device)
    
    # Cấu hình optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    # Training loop
    print("**************** Training started... ****************")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Chuyển dữ liệu lên device
            features = batch["features"].data.to(device)  # (1, T, D)
            feat_length = batch["feat_length"].data.to(device)
            targets = batch["tokens_eos"].data.to(device)  # (1, U+1)
            target_length = batch["target_length"].data.to(device)

            # print(f"\nFeatures shape: {features.data.size()}")
            # print(f"Features: {features}")
            # print(f"Feat length: {feat_length}")
            # print(f"\nTargets shape: {targets.data.size()}")
            # print(f"Targets: {targets}")
            # print(f"Target length: {target_length}")
            # exit()
            # Forward pass
            log_probs = model(
                inputs=features,
                input_lens=feat_length,
                targets=targets, 
                targets_lens=target_length,
            )
            # print(f"\nLogits: {log_probs}\n")
            # print(f"\nLogits shape: {log_probs.size()}")
            # print(f"\nTargets shape: {targets[:,:-1].contiguous().size()}")
            # print(f"\nTargets: {targets[:,:-1].to(torch.int32).contiguous()}")
            # print(f"\nLogit Length shape: {feat_length.size()}")
            # print(f"\nLogit Length : {feat_length}")
            # print(f"\nTarget length: {target_length-2}")
            # exit()

            # Tính loss

            loss = torchaudio.functional.rnnt_loss(
                logits=log_probs,  
                targets=targets[:,:-1].contiguous().to(torch.int32),
                logit_lengths=feat_length.to(torch.int32),
                target_lengths=(target_length-2).to(torch.int32)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Log mỗi 5 batch
            if batch_count % 5 == 0:
                print(f"[{datetime.datetime.now()}] Epoch {epoch} - Batch {batch_count}: Loss = {loss.item():.4f}")

        # Evaluation trên tập dev
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dev_batch in dev_loader:
                # Xử lý tương tự training
                features = dev_batch["features"].data.to(device)  # (1, T, D)
                feat_length = dev_batch["feat_length"].data.to(device)
                targets = dev_batch["tokens_eos"].data.to(device)  # (1, U+1)
                target_length = dev_batch["target_length"].data.to(device)

                log_probs = model(
                    inputs=features,
                    input_lens=feat_length,
                    targets=targets, 
                    targets_lens=target_length,
                )

                loss = torchaudio.functional.rnnt_loss(
                    logits=log_probs,  
                    targets=targets[:,:-1].contiguous().to(torch.int32),
                    logit_lengths=feat_length.to(torch.int32),
                    target_lengths=(target_length-2).to(torch.int32)
                )
                val_loss += loss.item()

        avg_train_loss = total_loss / batch_count
        avg_val_loss = val_loss / len(dev_dataset)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Lưu model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pt")
