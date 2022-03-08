import torch
import torch.nn.functional as F
import sys

# class DiceLoss(torch.nn.Module):
#     def __init__(self, n_classes=1, normalization='sigmoid', reduction='mean'):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#         self.reduction = reduction
#         self.normalization = normalization
#
#     def dice_loss(self, probs, ohe_labels):
#         # compute intersection
#         intersection = torch.sum(probs * ohe_labels, dim=1)
#         # compute union
#         union = torch.sum(probs * probs, dim=1) + torch.sum(ohe_labels * ohe_labels, dim=1)
#         dice_score = 2. * intersection / union
#         return 1 - dice_score
#
#     def forward(self, logits, labels):
#
#         if self.normalization == 'sigmoid':
#             dice = self.dice_loss(logits.sigmoid(), labels)
#         else: sys.exit('working on it')
#
#         if self.reduction == 'none':
#             return dice + torch.nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
#         elif self.reduction == 'mean':
#             return dice.mean()+ torch.nn.BCEWithLogitsLoss(reduction='mean')(logits, labels)
#         else:
#             raise ValueError('`reduction` must be \'none\' or \'mean\'.')

# def dice_loss(self, probs, cum_labels):
#     # compute intersection
#     intersection = torch.sum(probs * cum_labels, dim=1)
#     # compute union
#     union = torch.sum(probs * probs, dim=1) + torch.sum(cum_labels * cum_labels, dim=1)
#     dice_score = 2. * intersection / union
#     return 1 - dice_score

class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Adapted from https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, logits, target):
        assert logits.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = logits.sigmoid().contiguous().view(logits.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(predict * target, dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - intersection / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class BinaryAreaCEDiceLoss(torch.nn.Module):
    """
    """
    def __init__(self, smooth=1, p=1, reduction='mean'):
        super(BinaryAreaCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, logits, target):
        assert logits.shape[0] == target.shape[0], "predict & target batch size don't match"

        logits = logits.contiguous().view(logits.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(logits.sigmoid() * target, dim=1) + self.smooth
        union = torch.sum(logits.sigmoid().pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        dice_loss = 1 - intersection / union
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(dim=-1)
        area = target.sum(dim=-1).float()/target.shape[-1] # already flattened
        loss = (1-area)*dice_loss +area*bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))