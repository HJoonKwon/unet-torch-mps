# unet-torch-mps

## Installation 
TBD (requirements.txt)

## Folder structure 
```bash
 ❯ tree -L 2
.
├── __init__.py
├── config
│   └── default.yaml
├── dataset
│   ├── __init__.py
│   └── cityscapes.py
├── loss
│   ├── __init__.py
│   └── dice.py
├── metrics
│   ├── __init__.py
│   └── iou.py
├── model
│   ├── __init__.py
│   ├── blocks.py
│   └── unet.py
├── train
│   └── trainer.py
└── utils
    └── utils.py
```

## RUN 
TBD (scripts e.g. train, unit test, download dataset...)

## Result 
TBD (training curve, mIoU, training time on M2Air-8GB, ...)

## TODO 
- [ ] Refactor train.py in OOP way 
- [x] Update loss function to better one 
- [ ] Add unit test for dataset class
- [x] Load checkpoint and resume training
- [ ] Update readme 
- [ ] make docker file 
- [x] logger instead of print 
- [ ] wandb instead of terminal 
- [ ] torch compile
- [ ] implement scheduler 
