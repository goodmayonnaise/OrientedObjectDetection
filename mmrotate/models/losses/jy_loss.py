import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES

from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target, gamma, alpha, None,
                               'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



'''
16 channel = bg channel 
objectness thr 없이  그냥 확률을 곱해줌 
indicator function = gt가 fg일때만 동작 
indicator(focal(sig(pred cls * pred obj), gt))

version=1) objectness가 focal로 인한 학습에 영향을 안줘야 함 -> detach 
version=2) objectness가 두 loss를 통해 학습 
'''
@ROTATED_LOSSES.register_module()
class ObjectnessLoss2(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, obj_loss_weight=1.0, gamma=2.0, alpha=0.25, 
                 ver=1, **kwargs):
        super(ObjectnessLoss2, self).__init__()
        self.reduction = reduction 
        self.loss_weight = loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ver = ver 
        
        self.bce = nn.BCELoss()
        
    def forward(self, pred, target, 
                avg_factor=None, weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)        
        
        fg_mask = target!=15
        objectness = torch.sigmoid(pred[:,-1])
        loss_obj = self.obj_loss_weight * weight_reduce_loss(self.bce(objectness, fg_mask.float()), weight, reduction, avg_factor)
        
        if fg_mask.sum() == 0:
            loss_cls = torch.tensor(0., device=pred.device)
        else:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            fg_pred = pred[:, :-1][fg_mask]
            if self.ver==1:
                fg_pred = fg_pred*objectness[fg_mask].unsqueeze(1).detach()
            elif self.ver==2:
                fg_pred = fg_pred*objectness[fg_mask].unsqueeze(1)
                
            loss_cls = self.loss_weight * calculate_loss_func(fg_pred, target[fg_mask], 
                                                              weight, gamma=self.gamma, alpha=self.alpha, 
                                                              reduction=reduction, avg_factor=avg_factor)
            
        loss = loss_cls + loss_obj
        # print(f"loss_cls : {loss_cls} loss_obj : {loss_obj}")
        return loss 


'''
decoupled obj, cls head 
'''
@ROTATED_LOSSES.register_module()
class ObjectnessLoss3(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, obj_loss_weight=1.0, gamma=2.0, alpha=0.25, 
                 ver=1, **kwargs):
        super(ObjectnessLoss3, self).__init__()
        self.reduction = reduction 
        self.loss_weight = loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ver = ver
        
        self.bce = nn.BCELoss()
        
    def forward(self, cls_pred, obj_pred, target, 
                avg_factor=None, weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)        
        
        fg_mask = target!=15
        # objectness = torch.sigmoid(pred[:,-1])
        loss_obj = self.obj_loss_weight * weight_reduce_loss(self.bce(obj_pred.squeeze(), fg_mask.float()), weight, reduction, avg_factor)
        
        if fg_mask.sum() == 0:
            loss_cls = torch.tensor(0., device=cls_pred.device)
        else:
            if torch.cuda.is_available() and cls_pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            fg_pred = cls_pred[fg_mask]
            if self.ver==1:
                fg_pred = fg_pred*obj_pred[fg_mask].detach()
            elif self.ver==2:
                fg_pred = fg_pred*obj_pred[fg_mask]
                
            loss_cls = self.loss_weight * calculate_loss_func(fg_pred, target[fg_mask], 
                                                              weight, gamma=self.gamma, alpha=self.alpha, 
                                                              reduction=reduction, avg_factor=avg_factor)
            
        loss = loss_cls + loss_obj
        # print(f"loss_cls : {loss_cls} loss_obj : {loss_obj}")
        return loss 