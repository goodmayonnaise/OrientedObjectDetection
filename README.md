

</div>

## Introduction

JEONJIYOEN Oriented Object Detection CODE 

### Highlight

| Task                     | Dataset | AP                                   | 
| ------------------------ | ------- | ------------------------------------ | 
| base(RotatedYOLOv8)      | DOTA    | 72.8                                 | 
| Prototype1               | DOTA    | 78.0                                 | 
| Prototype3               | DOTA    | 79.3                                 | 
| Prototype4               | DOTA    | 79.5                                 |      


<div align=center>
<img src="https://github.com/user-attachments/assets/26c727d2-0f59-484f-8ebb-26f6a337b742"/>
</div>


- 내용1
- 내용2
- 내용3

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
# docker build 
docker run --name mmrotate_jyjeon --gpus all --shm-size=1024gb -it -v mmrotate_path/:/mmrotate -v /home/data/:/mmrotate/data -e TZ=Asia/Seoul mmrotate
```

```shell 
# additaional installation 
pip install gpustat
pip install typing_extentsions 
pip install einops
pip install tensorboard
pip install setuptools==59.5.0
pip install timm
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

