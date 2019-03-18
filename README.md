# TorchText TH
A (beta version) Thai word segmentation library built on PyTorch

## Usage
```python
>>> from typing import List
>>> from torchtext_th.tokenizer import get_tokenizer
>>> tokenizer = get_tokenizer("artifact/emb150_bilstm512_1layer.pt")
>>>
>>> input_text: str = "ลองทดสอบโปรแกรมตัดคำด้วย PyTorch ง่ายๆ จร้าาาา ถถถ"
>>> output: List[str] = tokenizer.tokenize(input_text)
>>> print("|".join(output))
ลอง|ทดสอบ|โปรแกรม|ตัด|คำ|ด้วย| |PyTorch| |ง่าย|ๆ| |จร้าาาา| |ถถถ
```
