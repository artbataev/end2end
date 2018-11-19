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
- [ ] CTC Greedy Decoder (C++, CPU)
- [ ] CTC Beam Search Decoder (C++, CPU)
- [ ] CTC Beam Search Decoder with language model (C++, CPU)

### Future:
- [ ] Gram-CTC Beam Search Decoder

## Requirements
- Pytorch 1.0 and higher
- numpy
- numba
