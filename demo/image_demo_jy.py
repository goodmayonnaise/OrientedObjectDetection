# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401
import time
import torch 

'''
config.py
pth.pth
'''

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--time', default=False, help='calculate time')
    args = parser.parse_args()
    return args


def main(args):
    t = args.time
    if t:
        device = torch.device(args.device)  # 문자열을 torch.device 객체로 변환

        if device.type == 'cuda':  # GPU 사용 시에만 VRAM 측정
            torch.cuda.reset_peak_memory_stats()  # GPU 메모리 초기화 (디바이스 문자열 사용)

        # 모델 로드 및 VRAM 측정
        start = time.perf_counter()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if t:
        end = time.perf_counter()
    
    if t:
        if device.type == 'cuda':
            vram_after_load = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB 단위
            vram_peak_after_load = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

            print(f"Model load time: {end - start:.4f} sec")
            print(f"VRAM after model load: {vram_after_load:.4f} GB")
            print(f"Peak VRAM after model load: {vram_peak_after_load:.4f} GB")

            torch.cuda.reset_peak_memory_stats()  # 추론 전 VRAM 초기화
    
    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    imgs_path = '/mmrotate/data/vis_sub_samples/vis_samples'
    imgs = os.listdir(imgs_path)

    for img in imgs: 
        out_file = os.path.join(('/').join(args.config.split('/')[:-1]), 'save/vis_sampels', img)
        # test a single image
        result = inference_detector(model, os.path.join(imgs_path, img))
        
        if t:
            if device.type == 'cuda':
                vram_after_inference = torch.cuda.memory_allocated(device) / (1024 ** 3)
                vram_peak_after_inference = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                print(f"Inference time: {end - start:.4f} sec")
                print(f"VRAM after inference: {vram_after_inference:.4f} GB")
                print(f"Peak VRAM after inference: {vram_peak_after_inference:.4f} GB")
            
        # show the results
        show_result_pyplot(
            model,
            os.path.join(imgs_path, img),
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=out_file)
            
        print('saved path :', out_file)



if __name__ == '__main__':
    args = parse_args()
    main(args)
