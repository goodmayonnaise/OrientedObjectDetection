

</div>

## Introduction

JEONJIYOEN Oriented Object Detection CODE 

### Highlight

| Task                     | Dataset | AP                                   | 
| ------------------------ | ------- | ------------------------------------ | 
| RotatedYOLOv8            | DOTA    | 72.8                                 | 
| Prototype3               | DOTA    | 79.3                                 | 
| Prototype4               | DOTA    | 79.8(single-scale)/81.3(multi-scale) | 


<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>


- 내용1
- 내용2
- 내용3

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Get Started

```shell
# train dist train 
tools/dist_train.sh configs/jy/prototype.py 

# inference
python demo/image_demo_jy.py work_dirs/config.py work_dirs/pth.pth 

# heatmap
python tools/heatmap.py 
python tools/heatmap/*.py 
```
