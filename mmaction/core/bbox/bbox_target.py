# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def bbox_target(pos_bboxes_list, neg_bboxes_list, gt_labels, cfg):
    """Generate classification targets for bboxes.

    Args:
        pos_bboxes_list (list[Tensor]): Positive bboxes list.pos_proposals
        neg_bboxes_list (list[Tensor]): Negative bboxes list. neg_proposals
        gt_labels (list[Tensor]): Groundtruth classification label list. pos_gt_labels
        cfg (Config): RCNN config.  rcnn_train_cfg
        cfg就是：我在config里面定义的rcnn文件。下面是一个例子、
        **rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),**

    Returns:
        (Tensor, Tensor): Label and label_weight for bboxes.
    """
    labels, label_weights = [], []
    pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight

    assert len(pos_bboxes_list) == len(neg_bboxes_list) == len(gt_labels)
    length = len(pos_bboxes_list)

    for i in range(length):
        pos_bboxes = pos_bboxes_list[i]
        neg_bboxes = neg_bboxes_list[i]
        gt_label = gt_labels[i]

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        label = F.pad(gt_label, (0, 0, 0, num_neg))
        label_weight = pos_bboxes.new_zeros(num_samples)
        label_weight[:num_pos] = pos_weight
        label_weight[-num_neg:] = 1.

        labels.append(label)
        label_weights.append(label_weight)

    labels = torch.cat(labels, 0)
    label_weights = torch.cat(label_weights, 0)
    return labels, label_weights
