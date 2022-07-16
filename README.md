# MemSR
MemSR: Training Memory-efficient Lightweight Model for Image Super-Resolution

## Introduction

This is the demo code for MemSR. Containing the main experiment in the paper. 

## Environments

* Codes are based on the Pytorch-Lightning framework.
* You can set up the environments as follows:
  1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 
  2. Create a new environment with `python=3.7`.
  3. Install packages in `requirements.txt`.
* To reproduce the main experiments, you need to download `DIV2K`, `Set5`, `Set14`, `B100`, and `Urban100` datasets. Check [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) to download the datasets.
* And place them to `/data/DIV2K`, `/data/Set5`, `/data/Set14`, and `/data/B100`, `/data/Urban100`. The dataset directory '/data' can be changed at `code/datasets/super_resolution/__init__.py:45`.
## Training

### Student Training

* `teacherx4_div2k_69068.ckpt` is a teacher model pretrained on DIV2K.
* Modify the `path_to_teacher` in `code/frameworks/distillation/start_jobs.py:8` to the path where the teacher model checkpoint.
* Run `python frameworks/distillation/start_jobs.py` to start the training.

## Acknowledgement

The code base is from [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) implementation, but it becomes very different from the code base now.
