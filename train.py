import os
import yaml
import torch
import json
import torchaudio
import datetime
from torch import nn
from torch.utils.data import DataLoader
from dataset import SpeechDataset, collate_fn
from model_builder import build_transformer_transducer
from prepare_data import get_transcript, create_vocab_file, create_word2index_file

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
    with open(r"config.yaml", 'r', encoding='utf-8') as f:
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
    vocab_folder = config["vocab_folder"]
    
    if os.path.exists(vocab_path):
        print(f"File vocab và các file w2i, i2w tồn tại!")
        w2i_dir = os.path.join(vocab_folder, 'word2index.json')
        i2w_dir = os.path.join(vocab_folder, 'index2word.json')
    else:
        print("Tiến hành tạo file vocab, w2i, i2w!") 
        data_file_list = [train_json, dev_json]
        transcript = get_transcript(data_file_list, 'processed_script')
        vocab_dir = create_vocab_file(transcript, vocab_path)
        w2i_dir, i2w_dir = create_word2index_file(vocab_folder, vocab_dir)

    with open(w2i_dir, 'r', encoding="utf-8") as w2i: 
        content_w2i = json.load(w2i)
        num_vocabs = len(content_w2i)
        blank_id = content_w2i.get("<blank>", None)

    with open(i2w_dir, 'r', encoding="utf-8") as w2i: 
        index2word = json.load(w2i)


    # Tạo Dataset và DataLoader cho train, dev, test
    train_dataset = SpeechDataset(
        json_path=train_json, 
        wav_folder=wav_folder, 
        vocab_path=w2i_dir, 
        sample_rate=sample_rate, 
        )
    dev_dataset = SpeechDataset(
        json_path=dev_json, 
        wav_folder=wav_folder, 
        vocab_path=w2i_dir, 
        sample_rate=sample_rate, 
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Thiết lập thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Khởi tạo mô hình Transformer-Transducer
    model = build_transformer_transducer(
        device=device, 
        num_vocabs=num_vocabs,
        input_size=80, 
        blank_id=blank_id
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
            # print(f"\nInputs length (Features length): {input_lens}")
            # print(f"\nTargets shape: {targets.shape}")
            # print(f"\nTargets: {targets}")
            # print(f"\nTarget length shape: {target_lens.shape}")
            # print(f"\nTarget length: {target_lens}")
            # exit()
            optimizer.zero_grad()
            # Forward: đưa qua mô hình để lấy log-prob dự đoán
            log_probs = model(inputs=inputs, 
                              input_lens=input_lens, 
                              targets=targets, 
                              targets_lens=target_lens
                              )

            # print("===================")
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
                            reduction='mean', 
                            blank=blank_id
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
                log_probs = model(inputs=inputs, 
                              input_lens=input_lens, 
                              targets=targets, 
                              targets_lens=target_lens
                              )
                
                loss = torchaudio.functional.rnnt_loss(
                            logits=log_probs,
                            targets=targets[:, 1:-1].to(torch.int32).contiguous(),
                            logit_lengths=input_lens.to(torch.int32),
                            target_lengths=(target_lens - 2).to(torch.int32),
                            reduction='mean',
                            blank=blank_id
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
            }, "/content/transformer_transducer_test/best_model.pth")
            print(f"*** Saved best model (epoch {epoch}, val_loss={val_loss:.4f}) ***")
        # Luôn lưu checkpoint mới nhất để có thể resume sau nếu cần
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, "/content/transformer_transducer_test/last_checkpoint.pth")
    
    print("**************** Huấn luyện hoàn tất! ****************")
    print("**************** Starting prediction... ****************")
    
    checkpoint = torch.load("best_model.pth")
    num_vocabs = len(index2word)

    # Khởi tạo mô hình
    model = build_transformer_transducer(device=device, num_vocabs=num_vocabs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Bắt đầu dự đoán trên tập huấn luyện...")

    # Vòng lặp dự đoán
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch_idx, (inputs, input_lens, targets, target_lens) in enumerate(train_loader):
            # Đưa dữ liệu lên GPU nếu có
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            # Dự đoán kết quả từ mô hình
            beam_size = 5  # Bạn có thể điều chỉnh beam_size tại đây
            y_hats = model.recognize(inputs, input_lens)

            # Lưu lại kết quả dự đoán và nhãn thực tế
            for pred, target in zip(y_hats, targets):
                # Chuyển đổi chỉ số từ vựng thành văn bản
                pred_sentence = [index2word[str(idx.item())] for idx in pred]
                target_sentence = [index2word[str(idx.item())] for idx in target if idx.item() != 0]

                predictions.append(" ".join(pred_sentence))
                ground_truths.append(" ".join(target_sentence))

            # In kết quả một vài batch để theo dõi
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}:")
                print(f"  Dự đoán: {' '.join(pred_sentence)}")
                print(f"  Nhãn thực: {' '.join(target_sentence)}")
                print()

    print("Dự đoán hoàn tất!")

    # Lưu kết quả vào file
    with open("train_predictions.txt", "w", encoding="utf-8") as f:
        for pred, gt in zip(predictions, ground_truths):
            f.write(f"Prediction: {pred}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write("\n")

    print("Kết quả dự đoán đã được lưu vào file train_predictions.txt")

    print("**************** Prediction complete! ****************")
    
if __name__ == "__main__":
    main()
