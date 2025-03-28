import os
import yaml
import torch
import json
import torchaudio
import datetime
from torch import nn
from torch.utils.data import DataLoader
from dataset import SpeechDataset, create_vocab_from_data, collate_fn
from model_builder import build_transformer_transducer
# Sử dụng hàm loss cho RNNT (cần cài đặt warp-transducer hoặc dùng torchaudio)
try:
    from torchaudio.functional import rnnt_loss  # warp-transducer loss
except ImportError:
    import torchaudio
    rnnt_criterion = lambda log_probs, targets, in_len, tgt_len: torchaudio.functional.rnnt_loss(
        log_probs, targets, in_len, tgt_len, blank=0, reduction='mean'
    )

def main():
    # Đọc file cấu hình
    with open(r"C:\paper\T-T\config.yaml", 'r', encoding='utf-8') as f:
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
    
    if not os.path.exists(vocab_path):
        vocab = create_vocab_from_data(train_json, vocab_path, use_subword)
        num_vocabs = len(vocab)  # Số lượng từ vựng là độ dài từ điển
        vocab = list(vocab.keys())  # Lấy danh sách các từ từ từ điển
    else:
        # Đọc vocab từ file JSON
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            vocab_dict = json.load(vf)
            num_vocabs = len(vocab_dict)  # Số lượng từ vựng là độ dài từ điển
            vocab = list(vocab_dict.keys())  # Lấy danh sách các từ từ từ điển

    # Tạo Dataset và DataLoader cho train, dev, test
    train_dataset = SpeechDataset(train_json, wav_folder, vocab, sample_rate, use_subword)
    dev_dataset = SpeechDataset(dev_json, wav_folder, vocab, sample_rate, use_subword)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Thiết lập thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Khởi tạo mô hình Transformer-Transducer
    model = build_transformer_transducer(
        device=device, num_vocabs=num_vocabs, input_size=train_dataset.n_mels
    )
    model.to(device)
    
    # Cấu hình optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    # Trainning stage
    print("**************** Training started... ****************")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, input_lens, targets, target_lens) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            # print(f"\nInputs (Features) shape: {inputs.shape}")
            # print(f"\nInputs length (Features length) shape: {input_lens.shape}")
            # print(f"\nTargets shape: {targets.shape}")
            # # print(f"\nTargets: {targets}")
            # print(f"\nTarget length shape: {target_lens.shape}")
            # # print(f"\nTarget length: {target_lens}")
            # # exit()
            optimizer.zero_grad()
            # Forward: đưa qua mô hình để lấy log-prob dự đoán
            log_probs = model(inputs=inputs, 
                              input_lens=input_lens, 
                              targets=targets, 
                              targets_lens=target_lens
                              )

            # print(f"\nLogits: {log_probs}\n")
            print("===================")
            # print(f"\nLogits shape: {log_probs.shape}")
            # print(f"\nTargets shape: {targets.contiguous().shape}")
            # print(f"\nTargets: {targets.to(torch.int32).contiguous()}")
            # print(f"\nLogit Length (input length) shape: {input_lens.shape}")
            # print(f"\nLogit Length (input length): {input_lens}")
            # print(f"\nTarget length: {target_lens}")
            # print(f"\nTarget length shape: {target_lens.shape}")
            # exit()
            # Tính loss RNNT (loại bỏ token <sos> ở đầu các chuỗi target trước khi tính)
            loss = torchaudio.functional.rnnt_loss(
                            logits=log_probs,
                            targets=targets[:, 1:-1].to(torch.int32).contiguous(),
                            logit_lengths=input_lens.to(torch.int32),
                            target_lengths=(target_lens - 2).to(torch.int32),
                            reduction='mean'
                        )

            loss_value = loss.item()
            total_loss += loss_value

            # Backpropagation
            loss.backward()
            
            # Cắt gradient để tránh gradient bùng nổ (nếu cần)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"[{datetime.datetime.now()}] Epoch {epoch} - Batch {batch_idx}: Loss = {loss_value:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"[{datetime.datetime.now()}] - [Epoch {epoch}] Training loss: {avg_loss:.4f}")
        
        # Đánh giá trên tập dev (tính loss trung bình)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, input_lens, targets, target_lens in dev_loader:
                inputs = inputs.to(device)
                input_lens = input_lens.to(device)
                targets = targets.to(device)
                target_lens = target_lens.to(device)
                # Forward trên tập dev
                log_probs = model(inputs, input_lens, targets, target_lens)
                # loss = rnnt_criterion(log_probs, targets[:, 1:], input_lens, target_lens - 1)
                loss = torchaudio.functional.rnnt_loss(
                            logits=log_probs,
                            targets=targets[:, 1:-1].to(torch.int32).contiguous(),
                            logit_lengths=input_lens.to(torch.int32),
                            target_lengths=(target_lens - 2).to(torch.int32),
                            reduction='mean'
                        )
                val_loss += loss.item()
        val_loss /= len(dev_loader)
        print(f"[{datetime.datetime.now()}] - [Epoch {epoch}] Validation loss: {val_loss:.4f}")
        
        # Lưu mô hình nếu cải thiện (loss giảm)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, "best_model.pth")
            print(f"*** Saved best model (epoch {epoch}, val_loss={val_loss:.4f}) ***")
        # Luôn lưu checkpoint mới nhất để có thể resume sau nếu cần
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, "last_checkpoint.pth")
    
    print("**************** Huấn luyện hoàn tất! ****************")
    
if __name__ == "__main__":
    main()
