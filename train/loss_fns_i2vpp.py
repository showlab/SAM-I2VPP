# -*- coding: utf-8 -*-
# @FileName: loss_fns_i2vpp.py
# @Time    : 2/5/25 16:19
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from trainer import CORE_LOSS_KEY

from utils.distributed import get_world_size, is_dist_avail_and_initialized


def structure_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the structure loss combining weighted BCE and weighted IoU losses.
    Args:
        inputs: A float tensor of shape [N, H, W] or [N, M, H, W].
                The predictions (logits) for each example.
        targets: A float tensor with the same shape as inputs.
                 Stores the binary classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch.
        loss_on_multimask: True if multimask prediction is enabled.
    Returns:
        Structure loss tensor of shape [N, M] if loss_on_multimask is True,
        otherwise a scalar tensor.
    """
    if loss_on_multimask:
        # inputs/targets shape is [N, M, H, W]
        N, M, H, W = targets.shape
        inputs = inputs.reshape(N * M, 1, H, W)
        targets = targets.reshape(N * M, 1, H, W)
    else:
        # inputs/targets shape is [N, H, W]
        N, H, W = targets.shape
        inputs = inputs.reshape(N, 1, H, W)
        targets = targets.reshape(N, 1, H, W)

    # compute 'weit'
    avg_pool = F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15)
    weit = 1 + 5 * torch.abs(avg_pool - targets)

    # compute 'wbce'
    wbce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    epsilon = 1e-6
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + epsilon)  # [N*M, 1]

    # compute 'wiou'
    pred = torch.sigmoid(inputs)
    inter = ((pred * targets) * weit).sum(dim=(2, 3))  # [N*M, 1]
    union = ((pred + targets) * weit).sum(dim=(2, 3))  # [N*M, 1]
    wiou = 1 - (inter + 1e-6) / (union - inter + 1e-6)  # [N*M, 1]

    # overall loss
    loss = wbce + wiou  # [N*M, 1]
    loss = loss.squeeze(-1)  # [N*M]

    if loss_on_multimask:
        # reshape to [N, M]
        loss = loss.view(N, M)
        assert loss.dim() == 2  # [N, M]
        assert num_objects > 0, "num_objects must be greater than 0"
        return loss / num_objects
    else:
        assert num_objects > 0, "num_objects must be greater than 0"
        loss = loss.sum() / num_objects
        return loss


def bceiou_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute unweighted BCE + IoU loss.

    Args:
        inputs: Predicted logits, shape [N, H, W] or [N, M, H, W]
        targets: Binary ground truth, same shape as inputs (0 or 1)
        num_objects: Number of objects in the batch, used for normalization
        loss_on_multimask: If True, inputs/targets have shape [N, M, H, W] and
                           the function returns a loss matrix of shape [N, M];
                           otherwise returns a scalar loss.

    Returns:
        If loss_on_multimask=True, returns a tensor of shape [N, M];
        otherwise returns a scalar tensor.
    """
    # Make the channel dimension equal to 1 for unified processing.
    if loss_on_multimask:
        N, M, H, W = targets.shape
        inputs = inputs.reshape(N * M, 1, H, W)
        targets = targets.reshape(N * M, 1, H, W)
    else:
        N, H, W = targets.shape
        inputs = inputs.reshape(N, 1, H, W)
        targets = targets.reshape(N, 1, H, W)

    eps = 1e-6
    pixel_count = H * W

    # 1. BCE term
    # reduction='none' keeps per-pixel losses, spatial averaging is done later
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    # Sum over spatial dimensions and normalize by the number of pixels, shape [N*M, 1]
    bce = bce.sum(dim=(2, 3)) / (pixel_count + eps)

    # 2. IoU term
    pred = torch.sigmoid(inputs)
    # Intersection
    inter = (pred * targets).sum(dim=(2, 3))
    # Union = pred + target - intersection
    union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    iou_loss = 1 - iou  # Shape [N*M, 1]

    # 3. Combine losses
    loss = bce + iou_loss  # Shape: [N*M, 1]
    loss = loss.squeeze(-1)  # Shape: [N*M]

    # 4. Output according to mode
    assert num_objects > 0, "num_objects must be a positive integer"
    if loss_on_multimask:
        loss = loss.view(N, M)
        return loss / num_objects  # Per-sample, per-mask loss matrix
    else:
        # Sum losses over all samples and normalize
        return loss.sum() / num_objects


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        memory_masks_list = outputs["multistep_pred_multimasks_memory"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(memory_masks_list) == len(ious_list), print(f"len(memory_masks_list) = {len(memory_masks_list)} while len(ious_list) = {len(ious_list)}")
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, memory_masks, ious, object_score_logits in zip(
            src_masks_list, memory_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, memory_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, memory_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # 1. compute src mask loss
        num_src_masks = src_masks.size(1)
        target_masks_src = target_masks.expand(-1, num_src_masks, -1, -1)
        loss_multimask_src = structure_loss(src_masks, target_masks_src, num_objects, loss_on_multimask=True)

        # 2. compute memory mask loss
        num_memory_masks = memory_masks.size(1)
        target_masks_memory = target_masks.expand(-1, num_memory_masks, -1, -1)
        loss_multimask_memory = bceiou_loss(memory_masks, target_masks_memory, num_objects, loss_on_multimask=True)
        loss_mask_memory = loss_multimask_memory.mean(dim=1, keepdim=True)  # [B, 1]

        # 3. compute object score loss
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask_src.dtype, device=loss_multimask_src.device
            )
            target_obj = torch.ones(
                loss_multimask_src.shape[0],
                1,
                dtype=loss_multimask_src.dtype,
                device=loss_multimask_src.device,
            )
        else:
            target_obj = torch.any((target_masks_src[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # 4. compute iou loss
        loss_multiiou = iou_loss(
            src_masks,
            target_masks_src,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask_src.dim() == 2
        assert loss_multimask_memory.dim() == 2
        assert loss_multiiou.dim() == 2

        # 5. For multi-mask predictions from src, select the mask with the minimum loss for gradient backpropagation,
        #    and add the loss of our memory mask
        if loss_multimask_src.size(1) > 1:
            # take the mask indices with the smallest structure loss for back propagation
            loss_combo = loss_multimask_src
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask_src = loss_combo[batch_inds, best_loss_inds].unsqueeze(1)

            # perform addition
            loss_mask = loss_mask_src + loss_mask_memory  # [B, 1] + [B, 1]
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask_src + loss_mask_memory
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
