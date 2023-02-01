import csv
import json
import os
from unittest import mock

import pytest
import torch
from torch import LongTensor

from speeq.utils import utils
from tests.conftest import masking_params_mark, mask_from_lens_mark

FILES_ENCODING = 'utf-8'


def test_load_json(char2id, tmp_path):
    """Tests the load functionality of utils.utils.load_json
    """
    file_path = os.path.join(tmp_path, 'file.json')
    with open(file_path, 'w', encoding=FILES_ENCODING) as f:
        json.dump(char2id, f)
    data = utils.load_json(file_path, encoding=FILES_ENCODING)
    assert data == char2id


def test_load_text(text, tmp_path):
    """Tests the load functionality of utils.utils.load_text
    """
    file_path = os.path.join(tmp_path, 'file.txt')
    with open(file_path, 'w', encoding=FILES_ENCODING) as f:
        f.write(text)
    data = utils.load_text(file_path, encoding=FILES_ENCODING)
    assert data == text


def test_save_json(char2id, tmp_path):
    """Tests the save functionality of utils.utils.save_json
    """
    file_path = os.path.join(tmp_path, 'file.json')
    utils.save_json(
        file_path=file_path, data=char2id, encoding=FILES_ENCODING
        )
    with open(file_path, 'r', encoding=FILES_ENCODING) as f:
        data = json.load(f)
    assert data == char2id


def test_save_text(text, tmp_path):
    """Tests the save functionality of utils.utils.save_text
    """
    file_path = os.path.join(tmp_path, 'file.txt')
    utils.save_text(
        file_path=file_path, data=text, encoding=FILES_ENCODING
        )
    with open(file_path, 'r', encoding=FILES_ENCODING) as f:
        data = f.read()
    assert data == text


@pytest.mark.parametrize(
    'sep', (',', '|', '\t')
)
def test_load_csv(dict_csv_data, tmp_path, sep):
    """Tests the load functionality of utils.utils.load_csv
    while using differnt seprator
    """
    file_path = os.path.join(tmp_path, 'file.csv')
    with open(file_path, 'w', encoding=FILES_ENCODING) as f:
        writer = csv.DictWriter(f, dict_csv_data[0].keys(), delimiter=sep)
        writer.writeheader()
        writer.writerows(dict_csv_data)
    data = utils.load_csv(
        file_path=file_path, encoding=FILES_ENCODING, sep=sep
        )
    assert dict_csv_data == data


@masking_params_mark
def test_get_pad_mask_shape(seq_len, pad_len):
    """Tests the masking shape functionality of utils.utils.get_pad_mask
    by ensuring that the result's shape is equal to seq_len + pad_len
    """
    mask = utils.get_pad_mask(seq_len=seq_len, pad_len=pad_len)
    assert mask.shape[0] == seq_len + pad_len


@masking_params_mark
def test_get_pad_mask_value(seq_len, pad_len):
    """Tests the values of the mask generated by utils.utils.get_pad_mask
    by ensuring the data side has all True while the padding all False
    """
    mask = utils.get_pad_mask(seq_len=seq_len, pad_len=pad_len)
    if seq_len == mask.shape[0]:
        assert True
        return
    for idx, item in enumerate(mask):
        if item.item() is False:
            assert idx == seq_len
            break


@pytest.mark.parametrize(
    ('seq_len', 'pad_len'),
    (
        (0, 1),
        (0, 5),
    )
)
def test_get_pad_mask_exception(seq_len, pad_len):
    """Tests if the function raises an error whenever the data size
    equal to 0.
    """
    with pytest.raises(ValueError):
        utils.get_pad_mask(seq_len=seq_len, pad_len=pad_len)


@pytest.mark.parametrize(
    ('result_len', 'pad_len', 'data_len', 'kernel_size', 'stride', 'expected'),
    (
        (7,  0, 10, 4, 1, 7),
        (10, 0, 10, 1, 1, 10),
        (10, 1, 9,  1, 1, 9),
        (10, 6, 4,  1, 1, 4),
        (9,  0, 10, 2, 1, 9),
        (9,  1, 9,  2, 1, 9),
        (9,  4, 6,  2, 1, 6),
        (6,  4, 6,  5, 1, 6),
        (2,  4, 6,  5, 2, 2),
        (5,  4, 6,  2, 2, 3),
        (4,  4, 6,  4, 2, 3),
        (3,  4, 6,  5, 2, 3),
    )
    )
def test_calc_data_len_int(
        result_len, pad_len, data_len, kernel_size, stride, expected
        ):
    """Tests the functionality of calc_data_len under integer
    pad_len and data_len.
    """
    assert expected == utils.calc_data_len(
        result_len=result_len,
        pad_len=pad_len,
        data_len=data_len,
        kernel_size=kernel_size,
        stride=stride
        )


@pytest.mark.parametrize(
    ('result_len', 'pad_len', 'data_len', 'kernel_size', 'stride', 'expected'),
    (
        (
            7,
            LongTensor([0]),
            LongTensor([10]),
            4,
            1,
            LongTensor([7])
        ),
        (
            10,
            LongTensor([0, 1, 6]),
            LongTensor([10, 9, 4]),
            1,
            1,
            LongTensor([10, 9, 4])
        ),
        (
            9,
            LongTensor([0, 1, 4]),
            LongTensor([10, 9, 6]),
            2,
            1,
            LongTensor([9, 9, 6])
        )
    )
    )
def test_calc_data_len_tensor(
        result_len, pad_len, data_len, kernel_size, stride, expected
        ):
    """Tests the functionality of calc_data_len under tensor
    pad_len and data_len.
    """
    mask = expected == utils.calc_data_len(
        result_len=result_len,
        pad_len=pad_len,
        data_len=data_len,
        kernel_size=kernel_size,
        stride=stride
        )
    assert mask.sum(dim=-1).item() == data_len.shape[0]


@pytest.mark.parametrize(
    ('result_len', 'pad_len', 'data_len', 'kernel_size', 'stride'),
    (
        (0, LongTensor([5]), 4, 0, 0),
        (0, 0, LongTensor([5]), 0, 0)
    )
)
def test_calc_data_len_invalid_types(
        result_len, pad_len, data_len, kernel_size, stride
        ):
    """Tests if the function calc_data_len raises an error whenever
    the passed parameters have invalid data types.
    """
    with pytest.raises(ValueError):
        utils.calc_data_len(
            result_len=result_len,
            pad_len=pad_len,
            data_len=data_len,
            kernel_size=kernel_size,
            stride=stride
            )


def test_get_positional_encoding_values(positional_enc_1_5_4):
    """Tests the correctness of the generated positional encoding values from
    get_positional_encoding function.
    """
    generated_pe = utils.get_positional_encoding(max_length=5, d_model=4)
    assert torch.allclose(
        generated_pe, positional_enc_1_5_4, rtol=1e-04, atol=1e-05
        )


@pytest.mark.parametrize(
    ('seq_len', 'd_model'),
    (
        (1, 10),
        (2, 8)
    )
)
def test_get_positional_encoding_shape(seq_len, d_model):
    """Tests the shape correctness of the generated positional
    encoding values from get_positional_encoding function.
    """
    shape = utils.get_positional_encoding(
        max_length=seq_len, d_model=d_model
        ).shape
    assert shape == (1, seq_len, d_model)


def test_get_positional_encoding_invalid_inp():
    """Tests if the function raises an exception if invalid input is passed.
    """
    with pytest.raises(ValueError):
        utils.get_positional_encoding(max_length=5, d_model=1)


@mask_from_lens_mark
def test_get_mask_from_lens_shape(lengths, max_len):
    """Tests the shape generated from get_mask_from_lens function.
    """
    mask = utils.get_mask_from_lens(lengths=lengths, max_len=max_len)
    assert mask.shape == (lengths.shape[0], max_len)


@mask_from_lens_mark
def test_get_mask_from_lens_sum_validity(lengths, max_len):
    """Tests the values of the generated mask of
    """
    mask = utils.get_mask_from_lens(lengths=lengths, max_len=max_len)
    assert (mask.sum(dim=-1) == lengths).sum().item() == lengths.shape[0]


def test_get_mask_from_lens_values():
    """Tests the mask values correctness generated from get_mask_from_lens
    """
    length = 5
    mask = utils.get_mask_from_lens(lengths=LongTensor([length]), max_len=7)
    mask = mask[0]
    for i, item in enumerate(mask):
        if item.item() is False:
            break
    assert i == length


@pytest.mark.parametrize(
    'batch_size',
    (1, 2, 3)
)
@mock.patch('speeq.utils.utils.get_positional_encoding')
def test_add_pos_enc_values(
        pos_mock, positional_enc_1_5_4, batched_speech_feat, batch_size
        ):
    """Test the functionality of adding in add_pos_enc by ensuring
    the positional encoding get added to the input
    """
    pos_mock.return_value = positional_enc_1_5_4
    batch = batched_speech_feat(batch_size, 5, 4)
    result = utils.add_pos_enc(batch)
    diff = result - batch
    assert torch.allclose(positional_enc_1_5_4, diff, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize(
    'batch_size',
    (1, 2, 3)
)
@mock.patch('speeq.utils.utils.get_positional_encoding')
def test_add_pos_enc_shape(
        pos_mock, positional_enc_1_5_4, batched_speech_feat, batch_size
        ):
    """Test the returned shape of add_pos_enc.
    """
    pos_mock.return_value = positional_enc_1_5_4
    shape = (batch_size, 5, 4)
    batch = batched_speech_feat(*shape)
    result = utils.add_pos_enc(batch)
    assert result.shape == shape
