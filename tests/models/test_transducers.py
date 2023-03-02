import pytest
import torch

from speeq.models import transducers
from tests.helpers import IGNORE_USERWARNING, check_grad, get_mask


class BaseTransducerTest:
    def check(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        enc_inp = batcher(batch_size, enc_len, enc_feat_size)
        dec_inp = int_batcher(batch_size, dec_len, n_classes)
        enc_mask = get_mask(seq_len=enc_len, pad_lens=enc_pad_lens)
        dec_mask = get_mask(seq_len=dec_len, pad_lens=dec_pad_lens)
        model = self.model(**model_args)
        result, speech_len, text_len = model(
            speech=enc_inp, speech_mask=enc_mask, text=dec_inp, text_mask=dec_mask
        )
        assert result.shape == expected_shape
        assert torch.all(expected_enc_lens == speech_len)
        assert torch.all(expected_dec_lens == text_len)
        check_grad(result=result, model=model)


class TestRNNTransducer(BaseTransducerTest):
    model = transducers.RNNTransducer

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
            "expected_enc_lens",
            "expected_dec_lens",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": False,
                    "rnn_type": "rnn",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 2,
                    "hidden_size": 12,
                    "bidirectional": False,
                    "rnn_type": "lstm",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": False,
                    "rnn_type": "lstm",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": False,
                    "rnn_type": "gru",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 2,
                    "hidden_size": 12,
                    "bidirectional": False,
                    "rnn_type": "gru",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "rnn",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 2,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "rnn",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "lstm",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 2,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "lstm",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "gru",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 3,
                    "n_dec_layers": 2,
                    "hidden_size": 12,
                    "bidirectional": True,
                    "rnn_type": "gru",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
            expected_enc_lens,
            expected_dec_lens,
        )


class TestConformerTransducer(BaseTransducerTest):
    model = transducers.ConformerTransducer

    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
            "expected_enc_lens",
            "expected_dec_lens",
        ),
        (
            (
                {
                    "d_model": 16,
                    "n_conf_layers": 2,
                    "n_dec_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "n_classes": 5,
                    "emb_dim": 8,
                    "rnn_type": "rnn",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "d_model": 16,
                    "n_conf_layers": 2,
                    "n_dec_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "n_classes": 5,
                    "emb_dim": 8,
                    "rnn_type": "lstm",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "d_model": 16,
                    "n_conf_layers": 2,
                    "n_dec_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "n_classes": 5,
                    "emb_dim": 8,
                    "rnn_type": "gru",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "d_model": 16,
                    "n_conf_layers": 2,
                    "n_dec_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 2,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "n_classes": 5,
                    "emb_dim": 8,
                    "rnn_type": "rnn",
                    "p_dropout": 0.1,
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 5, 4, 5),
                torch.LongTensor([5, 5]),
                torch.LongTensor([4, 2]),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
            expected_enc_lens,
            expected_dec_lens,
        )


class TestContextNet(BaseTransducerTest):
    model = transducers.ContextNet

    # @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
            "expected_enc_lens",
            "expected_dec_lens",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "n_sub_layers": 3,
                    "stride": 1,
                    "out_channels": [8, 16],
                    "kernel_size": 3,
                    "reduction_factor": 2,
                    "rnn_type": "rnn",
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 2,
                    "n_dec_layers": 2,
                    "n_sub_layers": 3,
                    "stride": 1,
                    "out_channels": [8, 16],
                    "kernel_size": 3,
                    "reduction_factor": 2,
                    "rnn_type": "rnn",
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 2,
                    "n_dec_layers": 2,
                    "n_sub_layers": 3,
                    "stride": 1,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "reduction_factor": 2,
                    "rnn_type": "rnn",
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 2,
                    "n_dec_layers": 2,
                    "n_sub_layers": 3,
                    "stride": 1,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "reduction_factor": 2,
                    "rnn_type": "lstm",
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "emb_dim": 4,
                    "n_layers": 2,
                    "n_dec_layers": 2,
                    "n_sub_layers": 3,
                    "stride": 1,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "reduction_factor": 2,
                    "rnn_type": "gru",
                },
                5,
                2,
                8,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 5),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
            expected_enc_lens,
            expected_dec_lens,
        )


class TestVGGTransformerTransducer(BaseTransducerTest):
    model = transducers.VGGTransformerTransducer

    # @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
            "expected_enc_lens",
            "expected_dec_lens",
        ),
        (
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "rnn_type": "rnn",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "rnn_type": "rnn",
                    "n_vgg_blocks": 2,
                    "n_conv_layers_per_vgg_block": [2, 2],
                    "kernel_sizes_per_vgg_block": [[2, 2], [2, 2]],
                    "n_channels_per_vgg_block": [[4, 8], [4, 4]],
                    "vgg_pooling_kernel_size": [2, 1],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "rnn_type": "lstm",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "rnn_type": "gru",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 3,
                    "rnn_type": "rnn",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 3,
                    "n_dec_layers": 1,
                    "rnn_type": "lstm",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "emb_dim": 6,
                    "n_layers": 2,
                    "n_dec_layers": 3,
                    "rnn_type": "gru",
                    "n_vgg_blocks": 1,
                    "n_conv_layers_per_vgg_block": [2],
                    "kernel_sizes_per_vgg_block": [[2, 2]],
                    "n_channels_per_vgg_block": [[4, 8]],
                    "vgg_pooling_kernel_size": [2],
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "left_size": 2,
                    "right_size": 3,
                    "p_dropout": 0.1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 2]),
                torch.LongTensor([4, 2]),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
            expected_enc_lens,
            expected_dec_lens,
        )


class TestTransformerTransducer(BaseTransducerTest):
    model = transducers.TransformerTransducer

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
            "expected_enc_lens",
            "expected_dec_lens",
        ),
        (
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "enc_left_size": 2,
                    "enc_right_size": 4,
                    "dec_left_size": 3,
                    "dec_right_size": 2,
                    "p_dropout": 0.01,
                    "stride": 1,
                    "kernel_size": 1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 6, 4, 7),
                torch.LongTensor([6, 5]),
                torch.LongTensor([4, 2]),
            ),
            (
                {
                    "in_features": 12,
                    "n_classes": 7,
                    "n_layers": 2,
                    "n_dec_layers": 1,
                    "d_model": 16,
                    "ff_size": 8,
                    "h": 4,
                    "joint_size": 8,
                    "enc_left_size": 2,
                    "enc_right_size": 4,
                    "dec_left_size": 3,
                    "dec_right_size": 2,
                    "p_dropout": 0.01,
                    "stride": 2,
                    "kernel_size": 1,
                },
                7,
                2,
                12,
                6,
                4,
                [0, 1],
                [0, 2],
                (2, 3, 4, 7),
                torch.LongTensor([3, 3]),
                torch.LongTensor([4, 2]),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
        expected_enc_lens,
        expected_dec_lens,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
            expected_enc_lens,
            expected_dec_lens,
        )
