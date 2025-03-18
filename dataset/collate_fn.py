import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features, tokens = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=0)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    return features, tokens
