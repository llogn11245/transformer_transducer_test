import yaml
import torch
from .dataset import SpeechDataset, collate_fn
from .model_builder import build_transformer_transducer

# Sử dụng thư viện jiwer để tính WER/CER
try:
    from jiwer import wer
except ImportError:
    raise ImportError("Please install jiwer: pip install jiwer")

def main():
    # Đọc cấu hình
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    sample_rate = config.get("sample_rate", 16000)
    use_subword = config.get("use_subword", False)
    vocab_path = config["vocab_path"]
    test_json = config["test_json"]
    wav_folder = config["wav_folder"]
    
    # Đọc vocab
    with open(vocab_path, 'r', encoding='utf-8') as vf:
        vocab = [line.strip() for line in vf]
    num_vocabs = len(vocab)
    
    # Tạo dataset và dataloader cho tập test (batch_size=1 để decode từng câu)
    test_dataset = SpeechDataset(test_json, wav_folder, vocab, sample_rate, use_subword)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Thiết lập thiết bị và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer_transducer(device=device, num_vocabs=num_vocabs, input_size=test_dataset.n_mels)
    model.to(device)
    
    # Load trọng số mô hình đã huấn luyện (checkpoint tốt nhất)
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Tạo từ điển index->token để phiên dịch kết quả
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    references = []
    hypotheses = []
    with torch.no_grad():
        for inputs, input_lens, targets, target_lens in test_loader:
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            # Thực hiện nhận dạng (greedy decode)  
            y_hats = model.recognize(inputs, input_lens)  # kết quả dạng chỉ số (shape [1, T_pred])
            pred_ids = y_hats[0].cpu().numpy().tolist()
            # Loại bỏ các token đặc biệt trong kết quả
            pred_tokens = []
            for idx in pred_ids:
                if idx == 0:   # <pad> hoặc blank (giả định blank dùng index 0)
                    continue
                if idx == 2:   # <eos>
                    break
                pred_tokens.append(idx2token[idx])
            # Chuyển danh sách token thành chuỗi kết quả
            if use_subword:
                # Nếu là subword, ghép các mảnh lại (loại bỏ ký hiệu '_' dùng cho khoảng trắng nếu có)
                pred_text = "".join(pred_tokens)
                # Thay thế ký tự đặc biệt của SentencePiece (ví dụ: '▁' thành khoảng trắng)
                pred_text = pred_text.replace("▁", " ").strip()
            else:
                pred_text = "".join(pred_tokens)
            # Lấy chuỗi transcript gốc (loại bỏ token sos/eos)
            target_ids = targets[0].numpy().tolist()
            ref_tokens = [idx2token[idx] for idx in target_ids if idx not in (0, 1, 2)]
            if use_subword:
                ref_text = "".join(ref_tokens).replace("▁", " ").strip()
            else:
                ref_text = "".join(ref_tokens)
            # Lưu lại để tính WER/CER
            hypotheses.append(pred_text)
            references.append(ref_text)
    
    # Tính toán WER và CER
    wer_score = wer(references, hypotheses)
    # Tính CER bằng cách tính WER trên từng ký tự (ghép mỗi ký tự thành một "từ")
    ref_chars = [" ".join(list(r)) for r in references]
    hyp_chars = [" ".join(list(h)) for h in hypotheses]
    cer_score = wer(ref_chars, hyp_chars)
    print(f"WER trên tập test: {wer_score:.3f}")
    print(f"CER trên tập test: {cer_score:.3f}")

if __name__ == "__main__":
    main()
