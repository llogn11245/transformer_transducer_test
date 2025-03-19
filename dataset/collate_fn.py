import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features = [item["features"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    tokens_bos = [item["tokens_bos"] for item in batch]
    tokens_eos = [item["tokens_eos"] for item in batch]
    feat_lengths = torch.tensor([item["feat_length"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    tokens_bos_padded = pad_sequence(tokens_bos, batch_first=True, padding_value=0)
    tokens_eos_padded = pad_sequence(tokens_eos, batch_first=True, padding_value=-1)

    return {
        "features": features_padded,
        "tokens": tokens_padded,
        "tokens_bos": tokens_bos_padded,
        "tokens_eos": tokens_eos_padded,
        "feat_length": feat_lengths,
        "target_length": target_lengths
    }
