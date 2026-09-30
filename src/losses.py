import torch
import torch.nn.functional as F

def dice(logits,target,eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*target).sum((2,3)); den=p.sum((2,3))+target.sum((2,3))
    return (1-(2*inter+eps)/(den+eps)).mean()

def total_loss(pred,t,w):
    rb=F.binary_cross_entropy_with_logits(pred["region"],t["region"]); rd=dice(pred["region"],t["region"])
    cb=F.binary_cross_entropy_with_logits(pred["centerline"],t["centerline"]); cd=dice(pred["centerline"],t["centerline"])
    bb=F.binary_cross_entropy_with_logits(pred["boundary"],t["boundary"])
    dl=F.smooth_l1_loss(torch.sigmoid(pred["distance"]),t["distance"])
    loss=w["region_bce"]*rb+w["region_dice"]*rd+w["center_bce"]*cb+w["center_dice"]*cd+w["boundary_bce"]*bb+w["distance"]*dl
    return loss
