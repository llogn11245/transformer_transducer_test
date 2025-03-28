import yaml
import torch
from dataset import SpeechDataset, collate_fn
from model_builder import build_transformer_transducer
import json
from torch.utils.data import DataLoader
import torchaudio
import datetime
from jiwer import wer, cer

def main():
    # Đọc file cấu hình
    with open(r"C:\paper\T-T\config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    batch_size = config["batch_size"]
    sample_rate = config.get("sample_rate", 16000)
    test_json = config["test_json"]
    wav_folder = config["wav_folder"]
    vocab_folder = config["vocab_folder"]
    
    # Đọc vocab
    w2i_dir = vocab_folder + "\\word2index.json"
    with open(w2i_dir, 'r', encoding="utf-8") as w2i: 
        content_w2i = json.load(w2i)
        num_vocabs = len(content_w2i)

    # Tạo từ điển index2word để chuyển từ chỉ số sang từ
    i2w_dir = vocab_folder + "\\index2word.json"
    with open(i2w_dir, 'r', encoding="utf-8") as i2w:
        index2word = json.load(i2w)

    # Tạo dataset và dataloader cho tập test
    test_dataset = SpeechDataset(
        json_path=test_json, 
        wav_folder=wav_folder, 
        vocab_path=w2i_dir, 
        sample_rate=sample_rate, 
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Thiết lập thiết bị và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer_transducer(
        device=device, 
        num_vocabs=num_vocabs, 
        input_size=80
    )
    model.to(device)

    # Load trọng số mô hình đã huấn luyện (checkpoint tốt nhất)
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Khởi tạo các biến để tính toán
    test_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, input_lens, targets, target_lens in test_loader:
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            # Forward trên tập test
            log_probs = model(
                inputs=inputs, 
                input_lens=input_lens, 
                targets=targets, 
                targets_lens=target_lens
            )
            
            # Tính loss
            loss = torchaudio.functional.rnnt_loss(
                logits=log_probs,
                targets=targets[:, 1:-1].to(torch.int32).contiguous(),
                logit_lengths=input_lens.to(torch.int32),
                target_lengths=(target_lens - 2).to(torch.int32),
                reduction='mean'
            )
            test_loss += loss.item()

            # Chuyển đổi log_probs sang dự đoán (Greedy Decoding)
            predictions = torch.argmax(log_probs, dim=-1)

            for pred, tgt in zip(predictions, targets):
                pred_text = " ".join([index2word[str(idx)] for idx in pred.tolist() if str(idx) in index2word])
                tgt_text = " ".join([index2word[str(idx)] for idx in tgt.tolist() if str(idx) in index2word])

                # Tính WER và CER
                batch_wer = wer(tgt_text, pred_text)
                batch_cer = cer(tgt_text, pred_text)

                total_wer += batch_wer
                total_cer += batch_cer
                total_samples += 1

    # Tính toán tổng hợp
    avg_wer = total_wer / total_samples
    avg_cer = total_cer / total_samples
    test_loss /= len(test_loader)

    # In kết quả cuối
    print(f"\n[RESULT] Test Loss: {test_loss:.4f}, Avg WER: {avg_wer:.4f}, Avg CER: {avg_cer:.4f}")

if __name__ == "__main__":
    main()
