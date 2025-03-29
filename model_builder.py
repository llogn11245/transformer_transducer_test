from audio_encoder import AudioEncoder
from label_encoder import LabelEncoder
from model import TransformerTransducer
import torch


def build_transformer_transducer(
        device: torch.device,
        num_vocabs: int,
        input_size: int = 80,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_audio_layers: int = 18,
        num_label_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        blank_id: int = 0
) -> TransformerTransducer:
    encoder = build_audio_encoder(
        device,
        input_size,
        model_dim,
        ff_dim,
        num_audio_layers,
        num_heads,
        dropout,
        max_len,
    )
    decoder = build_label_encoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_label_layers,
        num_heads,
        dropout,
        max_len,
        pad_id,
        sos_id,
        eos_id,
    )
    return TransformerTransducer(encoder, decoder, num_vocabs, model_dim << 1, model_dim, blank_id=blank_id).to(device)


def build_audio_encoder(
        device: torch.device,
        input_size: int = 80,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_audio_layers: int = 18,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
) -> AudioEncoder:
    return AudioEncoder(
        device,
        input_size,
        model_dim,
        ff_dim,
        num_audio_layers,
        num_heads,
        dropout,
        max_len,
    )


def build_label_encoder(
        device: torch.device,
        num_vocabs: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_label_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
) -> LabelEncoder:
    return LabelEncoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_label_layers,
        num_heads,
        dropout,
        max_len,
        pad_id,
        sos_id,
        eos_id,
    )
