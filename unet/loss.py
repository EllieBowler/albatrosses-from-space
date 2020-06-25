import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.count = 0

    def forward(self, prediction_gt, target_gt, inputs=None):
        # GT loss #
        targets_gt_onehot = torch.zeros_like(prediction_gt)
        targets_gt_onehot.scatter_(1, target_gt, 1.0)
        positive_targets_gt = torch.clamp((target_gt != 0).float().sum(), min=1.0)

        prob_gt = F.softmax(prediction_gt, dim=1)  # softmax over depth (class masks)

        target_prob_gt = prob_gt * targets_gt_onehot + (1 - prob_gt) * (1 - targets_gt_onehot)
        alpha_weight_gt = self.alpha * targets_gt_onehot + (1 - self.alpha) * (1 - targets_gt_onehot)
        weight_gt = (alpha_weight_gt * ((1 - target_prob_gt) ** self.gamma)).detach()

        losses_gt = F.binary_cross_entropy_with_logits(prediction_gt, targets_gt_onehot, weight=weight_gt,
                                                       reduce=False) / weight_gt.sum()

        return losses_gt.sum()
