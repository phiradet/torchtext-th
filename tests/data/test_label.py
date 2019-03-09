import torch

from torchtext_th.data.label import BMESLabel


def test_label_basic_transform():
    labels = [['B', 'M', 'E', 'S'],
              ['M', 'M', 'M'],
              ['B', 'E']]
    expected_val = [[1, 2, 3, 4],
                    [2, 2, 2, 0],
                    [1, 3, 0, 0]]
    expected = torch.tensor(expected_val)

    bmes = BMESLabel()
    result = bmes.transform(labels, max_len=4)
    assert torch.all(torch.eq(result, expected))
