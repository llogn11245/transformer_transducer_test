from typing import Tuple
from models.layers import CReLu, Conv1DLayers, RowConv1D
from torch import nn
from torch import Tensor


class DeepSpeechV1Encoder(nn.Module):
    """Builds the DeepSpeech encoder described in
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.
        hidden_size (int): The layers' hidden size.
        n_linear_layers (int): The number of feed-forward layers.
        bidirectional (bool): if the rnn is bidirectional or not.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            in_features: int,
            hidden_size: int,
            n_linear_layers: int,
            bidirectional: bool,
            max_clip_value: int,
            rnn_type: str,
            p_dropout: float
    ) -> None:
        super().__init__()
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    in_features=in_features if i == 0 else hidden_size,
                    out_features=hidden_size
                ),
                CReLu(
                    max_val=max_clip_value
                ),
                nn.Dropout(
                    p=p_dropout
                )
            )
            for i in range(n_linear_layers)
        ])
        from .registry import PACKED_RNN_REGISTRY
        self.rnn = PACKED_RNN_REGISTRY[rnn_type](
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(
            self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        # mask of shape [B, M] and True if there's no padding
        # x of shape [B, T, F]
        lengths = mask.sum(dim=-1)
        for layer in self.ff_layers:
            x = layer(x)
        out, _, lengths = self.rnn(x, lengths.cpu())
        if self.bidirectional is True:
            out = out[..., :self.hidden_size] + out[..., self.hidden_size:]
        out = self.crelu(self.fc(out))
        return out, lengths


class DeepSpeechV2Encoder(nn.Module):
    """Implements the deep speech 2 encoder
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.
        kernel_size (int): The convolution layers' kernel size.
        stride (int): The convolution layers' stride.
        in_features (int): The input/speech feature size.
        hidden_size (int): The layers' hidden size.
        bidirectional (bool): if the rnn is bidirectional or not.
        n_rnn (int): The number of RNN layers.
        n_linear_layers (int): The number of linear layers.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
        tau (int): The future context size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            n_conv: int,
            kernel_size: int,
            stride: int,
            in_features: int,
            hidden_size: int,
            bidirectional: bool,
            n_rnn: int,
            n_linear_layers: int,
            max_clip_value: int,
            rnn_type: str,
            tau: int,
            p_dropout: float
    ) -> None:
        super().__init__()
        self.conv = Conv1DLayers(
            in_size=in_features,
            out_size=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            n_layers=n_conv,
            p_dropout=p_dropout
        )
        from .registry import PACKED_RNN_REGISTRY
        self.rnns = nn.ModuleList(
            [
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional
                )
                for _ in range(n_rnn)
            ]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size
                )
                for _ in range(n_linear_layers)
            ]
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.context_conv = RowConv1D(
            tau=tau, hidden_size=hidden_size
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(
            self, x: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.conv(x, lengths)
        out = self.crelu(out)
        for layer in self.rnns:
            out, _, lengths = layer(
                out, lengths
            )
            if self.bidirectional is True:
                out = out[..., :self.hidden_size] +\
                    out[..., self.hidden_size:]
            out = self.crelu(out)
        out = self.context_conv(out)
        for layer in self.linear_layers:
            out = layer(out)
            out = self.crelu(out)
        return out, lengths
