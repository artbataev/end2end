# Losses and decoders for end-to-end Speech Recognition and Optical Character Recognition with PyTorch

[ ![Build](https://travis-ci.com/artbataev/end2end.svg?branch=master) ](https://travis-ci.com/artbataev/end2end)

The module focuses on experiments with CTC-loss (Connectionist Temporal Classification) and its modifications. 

Under active development.

## Documentation
[ artbataev.github.io/end2end/ ](https://artbataev.github.io/end2end/)

## Losses
- [x] CTC (C++, CPU)

## Decoders
- [x] CTC Greedy Decoder (C++, CPU)
- [x] CTC Beam Search Decoder (C++, CPU)
- [x] CTC Beam Search Decoder with language model (C++, CPU)

## How to install

**Requirements:**
- Python 3.4+
- PyTorch 1.0 and higher
- numpy
- numba

1. Install PyTorch 1.0 from [pytorch.org](https://pytorch.org), e.g.
    ```bash
    pip install torch
    ```

2. Install tools to compile
    ```bash
    sudo add-apt-repository ppa:ubuntu-toolchain-r-test
    sudo apt-get install cmake libboost-all-dev
    ```

3. Install the module
    ```bash
    pip install -v git+https://github.com/artbataev/end2end.git
    ```
    or
    ```bash
    git clone --recursive https://github.com/artbataev/end2end.git
    cd end2end
    python setup.py install
    python -m tests.test_ctc
    python -m tests.test_ctc_decoder
    ```

## How to use

### CTC Loss
```python
import torch
from pytorch_end2end import CTCLoss

ctc_loss = CTCLoss(blank_idx=0, time_major=False, 
    reduce=True, size_average=True, after_logsoftmax=False)

batch_size = 4
alphabet_size = 28 # blank + 26 english characters + space

logits = torch.randn(batch_size, 50, alphabet_size).detach().requires_grad_()
targets = torch.randint(1, alphabet_size, (batch_size, 30), dtype=torch.long)
logits_lengths = torch.full((batch_size,), 50, dtype=torch.long)
targets_lengths = torch.randint(10, 30, (batch_size,), dtype=torch.long)

loss = ctc_loss(logits, targets, logits_lengths, targets_lengths)
loss.backward()
```

### CTC Decoder
```python
import torch
from pytorch_end2end import CTCDecoder

batch_size = 4
alphabet_size = 6
decoder = CTCDecoder(blank_idx=0, beam_width=100, 
                     time_major=False, after_logsoftmax=False,
                     labels=["_", "a", "b", "c", "d", " "])

logits = torch.randn(batch_size, 50, alphabet_size).detach()
logits_lengths = torch.full((batch_size,), 50, dtype=torch.long)

decoded_targets, decoded_targets_lengths, decoded_sentences = decoder.decode(logits, logits_lengths)
for sentence in decoded_sentences:
    print(sentence)
```

### Future Plans

#### Losses
- [ ] Gram-CTC
- [ ] CTC without blank (C++, CPU)
- [ ] CTC (Cuda)
- [ ] CTC without blank refactoring
- [ ] Dynamic segmentation with CTC refactoring

#### Decoders
- [ ] Restrict Beam Search with vocabulary
- [ ] Allow custom transcriptions
- [ ] Gram-CTC Beam Search Decoder
