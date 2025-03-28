import yaml
import json
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer

from dataset import SpeechDataset, collate_fn
from model_builder import build_transformer_transducer

def main():
    # Đọc file cấu hình
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    batch_size = config.get("batch_size", 1)
    sample_rate = config.get("sample_rate", 16000)
    test_json = config["test_json"]
    wav_folder = config["wav_folder"]
    vocab_folder = config["vocab_folder"]

    # Tải từ điển word2index và index2word
    w2i_path = f"{vocab_folder}/word2index.json"
    i2w_path = f"{vocab_folder}/index2word.json"
    with open(w2i_path, "r", encoding="utf-8") as f:
        word2index = json.load(f)
    with open(i2w_path, "r", encoding="utf-8") as f:
        index2word = json.load(f)

    num_vocabs = len(word2index)
    sos_id = word2index.get("<sos>", 1)
    eos_id = word2index.get("<eos>", 2)
    pad_id = word2index.get("<pad>", 0)

    # Tạo dataset và dataloader cho tập test
    test_dataset = SpeechDataset(
        json_path=test_json,
        wav_folder=wav_folder,
        vocab_path=w2i_path,
        sample_rate=sample_rate
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Thiết lập thiết bị (GPU nếu có, nếu không thì CPU) và load mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer_transducer(device=device, num_vocabs=num_vocabs, input_size=80)
    model.to(device)
    # Load checkpoint đã huấn luyện
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Biến lưu trữ kết quả tổng hợp
    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0
    results = []  # danh sách lưu các cặp (target_text, pred_text)

    # Bắt đầu đánh giá mô hình trên tập test
    with torch.no_grad():
        for inputs, input_lens, targets, target_lens in test_loader:
            # Chuyển inputs sang device, độ dài input vẫn để trên CPU (hoặc cũng chuyển sang device tùy model)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)

            # Thực hiện suy luận (greedy decoding) bằng hàm recognize của mô hình
            predictions = model.recognize(inputs, input_lens)  # Tensor dự đoán dạng (B, T)

            # Duyệt qua từng mẫu trong batch để xử lý kết quả
            for i in range(predictions.size(0)):
                # Lấy chuỗi token dự đoán (trên CPU) và loại bỏ các token đặc biệt
                pred_seq = predictions[i].cpu().tolist()
                pred_tokens = [tok for tok in pred_seq if tok not in [sos_id, eos_id, pad_id]]
                # Ánh xạ các token thành từ (chuỗi ký tự) và ghép thành câu
                pred_text = " ".join([index2word[str(tok)] for tok in pred_tokens if str(tok) in index2word])

                # Lấy chuỗi token mục tiêu tương ứng và chuyển thành câu (loại bỏ token đặc biệt)
                tgt_seq = targets[i].tolist()  # targets hiện ở CPU (do DataLoader trả ra)
                tgt_tokens = [tok for tok in tgt_seq if tok not in [sos_id, eos_id, pad_id]]
                tgt_text = " ".join([index2word[str(tok)] for tok in tgt_tokens if str(tok) in index2word])

                # In câu dự đoán và câu mục tiêu
                print(f"Pred: {pred_text}")
                print(f"Target: {tgt_text}\n")

                # Tính WER và CER cho mẫu này, cộng dồn vào tổng
                total_wer += wer(tgt_text, pred_text)
                total_cer += cer(tgt_text, pred_text)
                total_samples += 1

                # Lưu kết quả vào danh sách
                results.append((tgt_text, pred_text))

    # Tính toán WER, CER trung bình trên toàn bộ tập test
    avg_wer = total_wer / total_samples if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    print(f"Test WER: {avg_wer * 100:.2f}%")
    print(f"Test CER: {avg_cer * 100:.2f}%")

    # Lưu tất cả các cặp Target - Prediction vào file văn bản
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for tgt_text, pred_text in results:
            f.write(f"Target: {tgt_text}\n")
            f.write(f"Prediction: {pred_text}\n\n")

if __name__ == "__main__":
    main()
