import torch

from torchtext_th.data.vocab import Vocab


def test_vocab_basic_fit_and_map():
    target_chars = ['A', 'B', 'C', 'D']
    vocab = Vocab().fit(target_chars)

    test_case = [['UNK', 'A', 'B', 'D', 'C'],
                 ['B', 'C']]
    result = vocab.transform(chars=test_case, max_len=6)
    expected = torch.tensor([[1, 2, 3, 5, 4, 0],
                             [3, 4, 0, 0, 0, 0]])
    assert torch.all(torch.eq(result, expected))


def test_vocab_basic_decode():
    target_chars = ['A', 'B', 'C', 'D']
    vocab = Vocab().fit(target_chars)

    expected = [['_UNK_', 'A', 'B', 'D', 'C'],
                ['B', 'C']]
    test_case = torch.tensor([[1, 2, 3, 5, 4, 0],
                              [3, 4, 0, 0, 0, 0]]).numpy()
    result = vocab.decode(test_case, ignore_pad=True)
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert expected[i][j] == result[i][j]


def test_vocab_get_target_chars():
    sentences = [['v0', 'v1', 'v2', 'v3', 'v0'],
                 ['v1', 'v3', 'v5'],
                 ['v1', 'v5', 'v0']]
    expected = {'v0', 'v1', 'v3', 'v5'}
    assert set(Vocab.get_unique_chars(sentences, min_freq=2)) == expected
