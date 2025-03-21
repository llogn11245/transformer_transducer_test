import torch
import torch.nn as nn
from models.encoder import AudioEncoder
from models.predictor import Predictor
from models.joint_network import JointNetwork

class TransformerTransducer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model, n_heads, ff_dim, enc_layers,
                 pred_embed_dim, pred_hidden_dim, pred_layers, joint_dim, dropout, blank_idx):
        super(TransformerTransducer, self).__init__()

        self.encoder = AudioEncoder(input_dim, d_model=d_model, n_heads=n_heads, ff_dim=ff_dim,
                               num_layers=enc_layers, dropout=dropout)

        self.predictor = Predictor(vocab_size, pred_embed_dim=pred_embed_dim, pred_hidden_dim=pred_hidden_dim,
                           num_layers=pred_layers, blank_idx=blank_idx)

        self.joint_network = JointNetwork(enc_dim=d_model, pred_dim=pred_hidden_dim, 
                                          joint_dim=joint_dim, vocab_size=vocab_size)

    def forward(self, audio_features, audio_lengths, target_sequence):
        enc_out = self.encoder(audio_features, audio_lengths)
        pred_out = self.predictor(target_sequence)
        logits = self.joint_network(enc_out, pred_out)
        return logits
