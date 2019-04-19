# LSUV.pytorch

Implementation of Layer-sequential unit-variance (LSUV) initialization which is proposed by [All you need is a good init](https://arxiv.org/abs/1511.06422) in PyTorch.

## Requirements

- PyTorch 1.0+

## How to use

```python
from lsuv import lsuv_init

model = lsuv_init(ResNet34(), train_loader, needed_std=1.0, std_tol=0.1,
                  max_attempts=10, do_orthonorm=True, device=device)
```

## Reference

- [Mishkin, Dmytro, and Jiri Matas. "All you need is a good init." arXiv preprint arXiv:1511.06422 (2015).](https://arxiv.org/abs/1511.06422)
