import os
import yaml
import json
import torchaudio
import datetime
from torch import nn
# from torch.utils.data import DataLoader
from model_builder import build_transformer_transducer
import torch
import speechbrain as sb 
from speechbrain.lobes.features import Fbank, MFCC

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

    """
    {
    'id': '0', 
    
    'features': tensor([[-10.5803,  -9.3644,  -7.4043,  ...,  -8.4489, -12.6506, -13.3423],
        [-18.9188, -11.1392,  -6.5608,  ..., -10.9439, -17.8307, -15.3159],
        [ -5.3786,  -3.3462,  -0.6602,  ...,  -9.9059, -15.6514, -15.9562],
        ...,
        [ -7.5914,  -3.4675,   0.3056,  ..., -11.0348, -18.8616, -16.2496],
        [-12.7942,  -5.1061,  -0.5397,  ..., -13.8752, -19.0713, -21.2390],
        [ -9.0233,  -8.1640,  -6.6176,  ..., -15.2897, -14.6032, -15.7705]]), 
        
    'encoded_text': tensor([ 151, 5584,  976,  816, 3039, 3714, 2094, 4614, 4272, 3722, 5804, 3143,
        1099,  720,  958, 1847, 5271, 2207, 2192, 4838, 4736, 3600, 3278, 2261,
        3214, 1159,  267,  831, 6356, 3655]), 
    
    'tokens_bos': tensor([   4,  151, 5584,  976,  816, 3039, 3714, 2094, 4614, 4272, 3722, 5804,
        3143, 1099,  720,  958, 1847, 5271, 2207, 2192, 4838, 4736, 3600, 3278,
        2261, 3214, 1159,  267,  831, 6356, 3655]), 
    
    'tokens_eos': tensor([ 151, 5584,  976,  816, 3039, 3714, 2094, 4614, 4272, 3722, 5804, 3143,
        1099,  720,  958, 1847, 5271, 2207, 2192, 4838, 4736, 3600, 3278, 2261,
        3214, 1159,  267,  831, 6356, 3655,    3]), 
    
    'feat_length': 827, 
    
    'target_length': 32
    }
    """

    train_dataset = get_dataset(train_json, False)
    dev_dataset = get_dataset(dev_json, False)
    test_dataset = get_dataset(test_json, False)
    
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

    # # Trainning stage
    # print("**************** Training started... ****************")
    # for epoch in range(1, num_epochs + 1):
    #     model.train()
    #     total_loss = 0.0
    #     for batch in train_dataset: 
    #         batch = batch.to(device)

    #         # inputs = inputs.to(device)
    #         # input_lens = input_lens.to(device)
    #         # targets = targets.to(device)
    #         # target_lens = target_lens.to(device)

    #         features = batch["features"]
    #         feat_lens = batch["feat_length"]
    #         targets = batch["tokens_eos"]  # Sử dụng tokens_eos (đã có eos_id)
    #         target_lens = batch["target_length"]

    #         optimizer.zero_grad()
    #         # Forward: đưa qua mô hình để lấy log-prob dự đoán
    #         # log_probs = model(inputs, input_lens, targets, target_lens)
    #         log_probs = model(
    #             inputs=features.transpose(1, 2),  # (B, D, T)
    #             input_lens=feat_lens,
    #             targets=targets[:, :-1],  # Bỏ eos_id cuối cùng
    #             targets_lens=target_lens-1,
    #         )

    #         # Tính loss RNNT (loại bỏ token <sos> ở đầu các chuỗi target trước khi tính)
    #         loss = torchaudio.functional.rnnt_loss(
    #                         log_probs=log_probs.transpose(0,1),
    #                         targets=targets[:, 1:-1].to(torch.int32).contiguous(),
    #                         target_lengths=(target_lens-2).to(torch.int32),
    #                         blank=0
    #                     )

    #         loss_value = loss.item()
    #         total_loss += loss_value

    #         # Backpropagation
    #         loss.backward()
            
    #         # Cắt gradient để tránh gradient bùng nổ (nếu cần)
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    #         optimizer.step()
            
    #         if batch % 5 == 0:
    #             print(f"[{datetime.datetime.now()}] Epoch {epoch} - Batch {batch}: Loss = {loss_value:.4f}")
        
    #     avg_loss = total_loss / len(train_dataset)
    #     print(f"[{datetime.datetime.now()}] - [Epoch {epoch}] Training loss: {avg_loss:.4f}")
        
    #     # Đánh giá trên tập dev (tính loss trung bình)
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for inputs, input_lens, targets, target_lens in dev_loader:
    #             inputs = inputs.to(device)
    #             input_lens = input_lens.to(device)
    #             targets = targets.to(device)
    #             target_lens = target_lens.to(device)
    #             # Forward trên tập dev
    #             log_probs = model(inputs, input_lens, targets, target_lens)
    #             # loss = rnnt_criterion(log_probs, targets[:, 1:], input_lens, target_lens - 1)
    #             loss = torchaudio.functional.rnnt_loss(
    #                         logits=log_probs,
    #                         targets=targets[:, 1:-1].to(torch.int32).contiguous(),
    #                         logit_lengths=input_lens.to(torch.int32),
    #                         target_lengths=(target_lens - 2).to(torch.int32),
    #                         blank=0,
    #                         reduction='mean'
    #                     )
    #             val_loss += loss.item()
    #     val_loss /= len(dev_loader)
    #     print(f"[{datetime.datetime.now()}] - [Epoch {epoch}] Validation loss: {val_loss:.4f}")
        
    #     # Lưu mô hình nếu cải thiện (loss giảm)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save({
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "epoch": epoch
    #         }, "best_model.pth")
    #         print(f"*** Saved best model (epoch {epoch}, val_loss={val_loss:.4f}) ***")
    #     # Luôn lưu checkpoint mới nhất để có thể resume sau nếu cần
    #     torch.save({
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epoch
    #     }, "last_checkpoint.pth")
    
    # print("**************** Huấn luyện hoàn tất! ****************")
    
    # Training loop
    print("**************** Training started... ****************")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch in train_dataset:
            # Chuyển dữ liệu lên device
            batch = batch.to(device)
            
            features = batch["features"].transpose(0, 1).unsqueeze(0)  # (1, D, T)
            feat_length = torch.tensor([batch["feat_length"]])
            targets = batch["tokens_eos"].unsqueeze(0)  # (1, U+1)
            target_length = torch.tensor([batch["target_length"]])

            # Forward pass
            log_probs = model(
                inputs=features,
                input_lens=feat_length,
                targets=targets[:, :-1],  # Bỏ eos ở cuối
                targets_lens=target_length-1,
            )

            # Tính loss
            loss = rnnt_criterion(
                logits=log_probs.transpose(0, 1),  # (T, N, C)
                targets=targets[:, 1:].int(),      # Bỏ sos ở đầu
                input_lengths=feat_length.int(),
                target_lengths=(target_length-2).int(), # Trừ sos và eos
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
            for dev_batch in dev_dataset:
                # Xử lý tương tự training
                features = dev_batch["features"].transpose(0, 1).unsqueeze(0).to(device)
                feat_length = torch.tensor([dev_batch["feat_length"]]).to(device)
                targets = dev_batch["tokens_eos"].unsqueeze(0).to(device)
                target_length = torch.tensor([dev_batch["target_length"]]).to(device)

                log_probs = model(
                    inputs=features,
                    input_lens=feat_length,
                    targets=targets[:, :-1],
                    targets_lens=target_length-1,
                )

                loss = rnnt_criterion(
                    logits=log_probs.transpose(0, 1),
                    targets=targets[:, 1:].int(),
                    input_lengths=feat_length.int(),
                    target_lengths=(target_length-2).int(),
                )
                val_loss += loss.item()

        avg_train_loss = total_loss / batch_count
        avg_val_loss = val_loss / len(dev_dataset)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Lưu model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pt")