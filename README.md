# Losses and decoders for end-to-end Speech Recognition and Optical Character Recognition with Pytorch

[ ![Build](https://travis-ci.com/artbataev/end2end.svg?branch=master) ](https://travis-ci.com/artbataev/end2end)

**[ Documentation ](https://artbataev.github.io/end2end/)**


## Losses
- [x] Connectionist temporal classification (CTC) (python / numba)
- [x] CTC (C++, CPU)
- [x] CTC without blank (python / numba)
- [x] Dynamic segmentation with CTC (python)

### Future:
- [ ] CTC without blank (C++, CPU)
- [ ] CTC (Cuda)
- [ ] Gram-CTC


## Decoders
- [x] CTC Greedy Decoder (C++, CPU)
- [x] CTC Beam Search Decoder (C++, CPU)
- [ ] CTC Beam Search Decoder with language model (C++, CPU)

### Future:
- [ ] Gram-CTC Beam Search Decoder

## Requirements
- Pytorch 1.0 and higher
- numpy
- numba

## How to install
1. Install Pytorch 1.0 (now pre-release preview) from [pytorch.org](https://pytorch.org), e.g.
    ```bash
    pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
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
ctc_loss = CTCLoss(blank_idx=0, time_major=False, reduce=True, size_average=True, after_logsoftmax=False)
batch_size, alphabet_size = 2, 27
log_probs = torch.randn(batch_size, 50 ,alphabet_size).detach().requires_grad_()
targets = torch.randint(1, alphabet_size, (batch_size, 30), dtype=torch.long)
input_lengths = torch.full((batch_size,), 50, dtype=torch.long)
target_lengths = torch.randint(10,30,(batch_size,), dtype=torch.long)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()
```
