import numpy as np

from torchtext_th.utils import pad_sequence


def test_pad_sequence_basic():
    in_seq = [1, 2, 3, 4, 5]
    expected = np.array([[1, 2], [3, 4], [5, 0]])
    output_list = list(pad_sequence(in_seq, max_length=2, overlap=0, pad_val=0))
    output = np.array(output_list)
    assert (output == expected).all()


def test_pad_sequence_return_len():
    in_seq = [1, 2, 3, 4, 5]
    expected_out = [[1, 2], [3, 4], [5, 0]]
    expected_len = [2, 2, 1]
    output = pad_sequence(in_seq, max_length=2, overlap=0, pad_val=0,
                          return_len=True)

    for i, (out, seq_len) in enumerate(output):
        expected_out_i = np.array(expected_out[i])
        output_i = np.array(out)
        assert (expected_out_i == output_i).all()
        assert expected_len[i] == seq_len
