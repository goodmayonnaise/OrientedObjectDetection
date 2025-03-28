import cv2, os
import torch
import numpy as np

from mmrotate.models import build_backbone, build_neck, build_head
from mmcv import Config 

import torch.nn.functional as F

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'BG')

def apply_heatmap_to_image_per_channel(feature_map, original_image, alpha=0.6):
    num_channels = feature_map.shape[1]  # C, H, W
    heatmaps = []

    # 원본 이미지 정규화
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 전체 feature_map의 최대값으로 정규화
    feature_map_max = feature_map.max()
    
    for i in range(num_channels):
        channel_map = feature_map[:, i, :, :].squeeze().cpu().detach().numpy()  # (H, W)
        
        # 정규화 (최대값 기준)
        channel_map = (channel_map / feature_map_max.item()) * 255
        channel_map = np.clip(channel_map, 0, 255).astype(np.uint8)

        # heatmap 생성
        heatmap = cv2.applyColorMap(channel_map, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # 원본과 합성
        combined_img = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)
        heatmaps.append(combined_img)
        
    return heatmaps  # 채널별 heatmap 리스트 반환
    
# def save_feat_img_per_channels(featmaps, original_img, save_path, type, compose=False):
#     combined_images = []

#     for i, featmap in enumerate(feature_maps):
            
    
def apply_heatmap_to_image(feature_map, original_image, alpha=0.6):
    feature_map_avg = feature_map.mean(dim=1).squeeze().cpu().detach().numpy()  # (1, H, W) -> (H, W)
        
    heatmap = cv2.normalize(feature_map_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    combined_img = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)

    return combined_img

def save_feat_img(imgs, org_img, save_path, type, compose=False): # compose : split save img
    combined_images = []
    for i, feature_map in enumerate(imgs):
        combined_image = apply_heatmap_to_image(feature_map, org_img)
        combined_images.append(combined_image)
    
    if compose:
        for i in range(len(imgs)):
            cv2.imwrite(os.path.join(save_path,type+f'_test_{i}.png'), combined_images[i])
    else: 
        h, w, _ = combined_images[0].shape
        _image = np.zeros((h, len(combined_images) * w, 3), dtype=np.uint8)
    for i in range(len(combined_images)): 
        _image[:, w*i:w*(i+1),:] = combined_images[i] 
        cv2.imwrite(os.path.join(save_path,type)+ '.png', _image)
    
    print(f"\n{type} done")
    print(os.path.join(save_path,type)+ '.png')

def data_load(data_path):
    img = cv2.imread(data_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 1024))
    img = img_resized.astype('float32') / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 
    return img_tensor, img

def hook_fn(module, input, output):
    feature_maps.append(output) 

def weight_keys(part):
    return {key.replace(part+'.', ''): value for key, value in weights['state_dict'].items() if key.startswith(part)}

def vis_attention_map(input_img_path, attns, save_path):
    img = cv2.imread(input_img_path)
    h, w, c = img.shape
    attn_maps = []
    for i, attn in enumerate(attns):
        # resize_ratio = 1  
        if i==0:
            resize_ratio = 16
        elif i==1:
            resize_ratio = 16//2
        else:
            resize_ratio = 16//4
        # resize_ratio = 8//(i+1) 
        # resize_ratio = 1024//attn.shape[2]
        attn_avg = attn.mean(dim=1)[0]
        # attn_avg = attn[0,7]
        query_patch_idx = attn_avg.shape[1] // 2
        attention_map = attn_avg[:, query_patch_idx]
        # h_p, w_p = 1024 // attn.shape[-1], 1024 // attn.shape[-1]
        
        attention_2d = attention_map.reshape(resize_ratio, resize_ratio)
        attention_resize = F.interpolate(attention_2d.unsqueeze(0).unsqueeze(0),
                                        size = img.shape[:2], 
                                        mode='bilinear',
                                        align_corners=False).squeeze()
        
        # norm 
        attention_resize = (attention_resize - attention_resize.min()) / (attention_resize.max() - attention_resize.min())
        attention_resize = (attention_resize * 255).detach().cpu().numpy().astype(np.uint8)
        attention_colormap = cv2.applyColorMap(attention_resize, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, attention_colormap, 0.4, 0)
        attn_maps.append(overlay)
        
    combine_img = np.zeros((h, w*(len(attns)), 3), dtype=np.uint8)
    for i, attn in enumerate(attn_maps):
        combine_img[:, w*i:w*(i+1),:] = attn
    cv2.imwrite(save_path, combine_img)
    
    print('\nattention map done')
    print(save_path)
        
    # attn_avg = attn.mean(dim=1)[0]
    # query_patch_idx = attn_avg.shape[1] // 2
    # attention_map = attn_avg[:, query_patch_idx]
    # # h_p, w_p = 1024 // attn.shape[-1], 1024 // attn.shape[-1]
    
    # attention_2d = attention_map.reshape(resize_ratio, resize_ratio)
    # attention_resize = F.interpolate(attention_2d.unsqueeze(0).unsqueeze(0),
    #                                  size = img.shape[:2], 
    #                                  mode='bilinear',
    #                                  align_corners=False).squeeze()
    
    # # norm 
    # attention_resize = (attention_resize - attention_resize.min()) / (attention_resize.max() - attention_resize.min())
    # attention_resize = (attention_resize * 255).detach().cpu().numpy().astype(np.uint8)
    # attention_colormap = cv2.applyColorMap(attention_resize, cv2.COLORMAP_JET)
    
    # overlay = cv2.addWeighted(img, 0.6, attention_colormap, 0.4, 0)
    
    # cv2.imwrite(save_path, overlay)

if __name__ == '__main__':
    # input
    # small 
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P0130__819__0___41.png'
    # jh 
    # sample_data_path = '/mmrotate/data/dummy_vis/area/GTF4_P2011__1024__0___0.png'
    # jg 
    sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1512__1365__0___376.png'
    # large plane
    # sample_data_path = '/mmrotate/data/split_ms_dota2_2/val/images/P1023__1365__2097___0.png' # harbor ship
    # sample_data_path = '/mmrotate/data/vis_sub_samples/vis_samples/P1397__1024__1048___2096.png' # heli, plane
    

    # vis sub
    # sample_data_path = '/mmrotate/data/vis_sub_samples/bbox_label/P0007__682__808___0.png'
    
    # org_pretrained = '/mmrotate/work_dirs/softfocal-validmask/best_mAP_epoch_49.pth'
    # config_path = '/mmrotate/work_dirs/softfocal-validmask/softfocal.py'
    # org_pretrained = '/mmrotate/work_dirs/ce-softmax/best_mAP_epoch_48.pth'
    # config_path = '/mmrotate/work_dirs/ce-softmax/test.py'
    # org_pretrained = '/mmrotate/work_dirs/_done/base-focal-with-bg/epoch_19.pth'
    # config_path = '/mmrotate/work_dirs/_done/base-focal-with-bg/test.py'
    # prototype4 bg loss 
    # org_pretrained = '/mmrotate/work_dirs/cspnext-yolov8-bgloss/best_mAP_epoch_8.pth'
    # config_path = '/mmrotate/work_dirs/cspnext-yolov8-bgloss/cspnext-yolov8-bgloss.py'
    # org_pretrained = '/mmrotate/configs/jy/pretrained/prototype4-10_bestmAP795_45ep.pth'
    # config_path = '/mmrotate/work_dirs/_done/prototype4-10/prototype4.py'
    
    # org_pretrained = '/mmrotate/work_dirs/objectness3-ver1/best_mAP_epoch_2.pth'
    # config_path = '/mmrotate/work_dirs/objectness3-ver1/objectness3.py'
    
    # org_pretrained = '/mmrotate/work_dirs/objectness2-ver2/best_mAP_epoch_12.pth'
    # config_path = '/mmrotate/work_dirs/objectness2-ver2/objectness-ver2.py'
    
    # org_pretrained = '/mmrotate/work_dirs/objectness4-ver1/best_mAP_epoch_3.pth'
    # config_path = '/mmrotate/work_dirs/objectness4-ver1/objectness4-ver1.py'
    
    org_pretrained = '/mmrotate/work_dirs/objectness2-ver2/best_mAP_epoch_10.pth'
    config_path = '/mmrotate/work_dirs/objectness2-ver2/objectness-ver2.py'
    
    weights = torch.load(org_pretrained)

    save_path = os.path.join('/',*org_pretrained.split('/')[:-1], 'save', 'vis_featmap', org_pretrained.split('/')[-1].split('.')[0], sample_data_path.split('/')[-1].split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        
    cfg = Config.fromfile(config_path)
        
    img_tensor, img = data_load(sample_data_path)
    
    # backbone setting
    backbone_weights = weight_keys('backbone')
    backbone = build_backbone(cfg.model.backbone)
    backbone.eval()

    # neck setting
    if cfg.model.neck is not None:
        neck_weights = weight_keys('neck')
        neck = build_neck(cfg.model.neck)
        neck.eval()
    
    # head_setting
    head_weights = weight_keys('bbox_head')
    head = build_head(cfg.model.bbox_head)
    head.eval()
    
    feature_maps=[] 
    ## backbone
    backbone.load_state_dict(backbone_weights) 
        
    # feat map list 
    out_org = backbone(img_tensor) 
    out = list(out_org) 
    
    # stage 1 feature 따로 추가하는 방법 
    # out.insert(0,feature_maps[0])

    # save_feat_img(out, img, backbone_save_path)
    save_feat_img(out, img, save_path, type='backbone')
    
    # neck
    if cfg.model.neck is not None:
        neck.load_state_dict(neck_weights)
        # layers = [neck.top_down_layers[0].main_conv, neck.top_down_layers[1].main_conv, neck.bottom_up_layers[0].main_conv, neck.bottom_up_layers[1].main_conv] # 훅은 꼭 weight load한 후에 걸어야함
        # for l in layers:
        #     layer_to_hook = l
        #     hook = layer_to_hook.register_forward_hook(hook_fn)
        neck_out = neck(out_org)
        save_feat_img(neck_out, img, save_path, type='neck')
        
    
    ## head ----------------------------------------------------------------

    head.load_state_dict(head_weights)
    if cfg.model.neck is not None:
        head_out = head(neck_out)
    else:
        head_out = head(out_org)

    # if len(head_out)==4:
    #     head_out = head_out[1:]
    
    head_cls = ['head_cls', 'head_bbox', 'head_angle', 'head_obj']
    # head_cls = ['head_cls', 'head_bbox', 'head_angle']
    
    save_feat_img([i[:,:15] for i in head_out[0]], img, save_path, type=f"head_cls_fg")
    save_feat_img([i[:,15:] for i in head_out[0]], img, save_path, type=f"head_cls_obj")
    for i,o in enumerate(head_out):
        save_feat_img(o, img, save_path, type=f"{head_cls[i]}")
        
        
    # per channels start ------------------------------------------------------------- 
    # channel_path = os.path.join(save_path, 'per_channels')
    # if not os.path.exists(channel_path):
    #     os.makedirs(channel_path)    
    # a = [apply_heatmap_to_image_per_channel(head_out[0][i], img) for i in range(3)]
    
    # for n, (i, j, k) in enumerate(zip(a[0], a[1], a[2])):
    #     save_path_n = os.path.join(channel_path, f"{CLASSES[n]}.png")
    #     cv2.imwrite(save_path_n, np.concatenate([i, j, k], 1))
    #     print(f"done {save_path_n}")
        
    # per channel end ----------------------------------------------------------------
    
    # [cv2.imwrite(f"test{n}.png", np.concatenate([i, j, k], 1)) for n, (i,j,k) in enumerate(zip(a[0], a[1], a[2]))]
    

    # save_feat_img(([i[:,:15] for i in head_out[0]]), img, save_path, type=f"head_cls_class")
    # save_feat_img(([i[:,15:] for i in head_out[0]]), img, save_path, type=f"head_cls_bg")
    # save_feat_img(([-i[:,15:] for i in head_out[0]]), img, save_path, type=f"head_cls_bg_rev")

    # gamma = [1, 0.75, 0.5, 0.25]
    # for g in gamma:
    #     save_feat_img(([i[:,:15]-i[:,15:]*g for i in head_out[0]]), img, save_path, type=f"head_cls_class_test{g}")

    # for j in range(15):
    #     save_feat_img(([i[:,j:j+1,:,:] for i in head_out[0]]), img, save_path, type=f"head_cls_{CLASSES[j]}")
    # save_feat_img(([i[:,15:] for i in head_out[0]]), img, save_path, type=f"head_cls_15")
    # [save_feat_img(([i[:,i,:,:] for i in head_out[0]]), img, save_path, type=f"head_cls_bg")]
    # start add layer 
    # i = 2
    # layers = [head.cls_preds[i][0].decode1, head.cls_preds[i][0].decode2, head.cls_preds[i][0].decode3]
    # layers = [head.cls_preds[i][0].attend for i in range(3)]
    # layers = [head.cls_preds[0][2].mha.mha0.attn_drop]
    
    # layers = []
    # for i in range(3):
    #     for j in range(3):
    #         layers.append(head.cls_preds[i][2].mha[j].attn_drop)
            
    # for layer in layers:
    #     hook = layer.register_forward_hook(hook_fn)
        
    # # # end add layer 
        
    # head.load_state_dict(head_weights)
    
    # if cfg.model.neck is not None:
    #     head_out = head(neck_out)
    # else:
    #     head_out = head(out_org)


    # vis_attention_map(input_img_path=sample_data_path, 
    #                   attns=feature_maps,
    #                   save_path=os.path.join(save_path, 'attn.png'))
    # hook.remove()
    # feature_maps = []
    
    # for i in range(3):
    #     # head.cls_preds[i][1].mha.mha0.attend.register_forward_hook(hook_fn)
    #     # head.cls_preds[i][1].mlp.register_forward_hook(hook_fn)
    #     head.cls_preds[i][1].mlp[0].fc1.register_forward_hook(hook_fn)
    #     head.cls_preds[i][1].mlp[0].act.register_forward_hook(hook_fn)
    #     head.cls_preds[i][1].mlp[0].fc2.register_forward_hook(hook_fn)
    # i = 2
    # head.cls_preds[i][2].pixelshuffle.register_forward_hook(hook_fn)
    # head.cls_preds[i][2].decode1.register_forward_hook(hook_fn)
    # head.cls_preds[i][2].decode2.register_forward_hook(hook_fn)
    # head.cls_preds[i][2].decode3.register_forward_hook(hook_fn)
    # # head.cls_preds[i][2].decode4.register_forward_hook(hook_fn)
    # head.cls_preds[i][2].conv1x1.register_forward_hook(hook_fn)
    # for i in range(3):    
    #     head.cls_preds[i][0].ChannelGate.register_forward_hook(hook_fn)
    #     head.cls_preds[i][0].SpatialGate.register_forward_hook(hook_fn)
    #     head.cls_preds[i][3].decode1.register_forward_hook(hook_fn)
    #     head.cls_preds[i][3].decode2.register_forward_hook(hook_fn)
    #     head.cls_preds[i][3].decode3.register_forward_hook(hook_fn)
    # for i in range(3):
    #     layers.append(head.cls_preds[i][0].SpatialGate.spatial.bn)

    # for layer in layers:
    #     hook = layer.register_forward_hook(hook_fn)
        
    # # end add layer 
        
    # head.load_state_dict(head_weights)
    
    # if cfg.model.neck is not None:
    #     head_out = head(out_org)
    # else:
    #     head_out = head(out_org)
        
    # # vis_attention_map(input_img_path=sample_data_path, 
    # #                   attns=feature_maps,
    # #                   save_path=os.path.join(save_path, 'mlp.png'))
    # # save_feat_img(feature_maps, img, save_path, type=f'mlps')
    # save_feat_img(feature_maps, img, save_path, type=f'backboneout_p{i+3}_decode')
    # save_feat_img(feature_maps, img, save_path, type=f'attn1')
    # save_feat_img(feature_maps[:5], img, save_path, type=f'p3')
    # save_feat_img(feature_maps[5:10], img, save_path, type=f'p4')
    # save_feat_img(feature_maps[10:], img, save_path, type=f'p5')
    
    # hook.remove()
    # feature_maps = []    
    
    # layers = []
    # for i in range(3):
    #     layers.append(head.cls_preds[i][0].ChannelGate.mlp[-1])

    # for layer in layers:
    #     hook = layer.register_forward_hook(hook_fn)
        
    # # # end add layer 
        
    # head.load_state_dict(head_weights)
    
    # if cfg.model.neck is not None:
    #     head_out = head(neck_out)
    # else:
    #     head_out = head(out_org)

    # save_feat_img(feature_maps, img, save_path, type=f'cbam_channel')
    