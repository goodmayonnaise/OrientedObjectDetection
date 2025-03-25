from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmrotate  # noqa: F401
import torch
import mmcv
import cv2
import numpy as np
import pdb 
from datetime import datetime
def count_bboxes(detection_model,result, score_thr=0.3):
    num_classes = len(result)  # result는 클래스별 리스트
    class_counts = {}

    for i in range(num_classes):
        bboxes = result[i]  # 각 클래스별 바운딩 박스 (N, 5) [x1, y1, x2, y2, score]
        if len(bboxes) > 0:
            # 신뢰도 임계값 적용하여 필터링
            valid_bboxes = bboxes[bboxes[:, -1] >= score_thr]
            if len(valid_bboxes) > 0:
                class_counts[detection_model.CLASSES[i]] = len(valid_bboxes)
    
    return class_counts

def infer(model_weight,CONFIG,image,save_path) :
    detection_model = init_detector(CONFIG, model_weight,device='cpu')
    result = inference_detector(detection_model, image)    
    score_thr = 0.3  # 신뢰도 임계값

    show_result_pyplot(detection_model, image, result, score_thr=score_thr, palette='dota', out_file=save_path)
    class_counts = count_bboxes(detection_model,result,score_thr)
    print(class_counts)

    return result

if __name__ == '__main__':
    
    
    image = '/home/jhcho/work/code/gradio_service/mmrotate/mmrotate/images/P0221__1024__0___1172.png' # image path
    main_path = '/home/jhcho/work/code/mmroate/case3/prototype'

    # ##Case4 
    model_weight = f'{main_path}/prototype4.pth'
    CONFIG = f'{main_path}/prototype4.py'
    save_path = '/home/jhcho/work/code/mmroate/result_pro1.jpg'
    
    infer(model_weight,CONFIG,image,save_path)
