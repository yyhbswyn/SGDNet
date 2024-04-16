# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import AssignResult, MaxIoUAssigner  #调用AssignResult方法，和原始的MaxIoUAssigner
    from mmdet.core.bbox.builder import BBOX_ASSIGNERS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @BBOX_ASSIGNERS.register_module()
    class MaxIoUAssignerAVA(MaxIoUAssigner):
        """Assign a corresponding gt bbox or background to each bbox.

        Each proposals will be assigned with `-1`, `0`, or a positive integer
        indicating the ground truth index.

        - -1: don't care
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt

        Args:
            pos_iou_thr (float): IoU threshold for positive bboxes.
            neg_iou_thr (float | tuple): IoU threshold for negative bboxes.
            min_pos_iou (float): Minimum iou for a bbox to be considered as a
                positive bbox. Positive samples can have smaller IoU than
                pos_iou_thr due to the 4th step (assign max IoU sample to each
                gt). Default: 0.
            gt_max_assign_all (bool): Whether to assign all bboxes with the
                same highest overlap with some gt to that gt. Default: True.
        """

        # 重写方法，用来处理不是int型的gt_labels.这里传进来的overlaps是从
        def assign_wrt_overlaps(self, overlaps, gt_labels=None):
            """Assign w.r.t. the overlaps of bboxes with gts.

            Args:
                overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                    shape(k, n)计算的是二者的重合情况。.
                gt_labels (Tensor, optional): Labels of k gt_bboxes, shape
                    (k, ).

            Returns:
                :obj:`AssignResult`: The assign result.
            """
            #batchsize等于几，一批就进入几次这个方法里，这个方法走完，完成了bbox_forward_train
            num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)   #4，1

            # 1. assign -1 by default 给所有的bbox分配一个默认的值-1，这里只有1个bbox，所以就分了1个
            assigned_gt_inds = overlaps.new_full((num_bboxes, ),  #形状是1，-1填充
                                                 -1,
                                                 dtype=torch.long)

            if num_gts == 0 or num_bboxes == 0:
                # No ground truth or boxes, return empty assignment
                max_overlaps = overlaps.new_zeros((num_bboxes, ))
                if num_gts == 0:
                    # No truth, assign everything to background
                    assigned_gt_inds[:] = 0
                if gt_labels is None:
                    assigned_labels = None
                else:
                    assigned_labels = overlaps.new_full((num_bboxes, ),
                                                        -1,
                                                        dtype=torch.long)
                return AssignResult(
                    num_gts,
                    assigned_gt_inds,
                    max_overlaps,
                    labels=assigned_labels)

            # for each anchor, which gt best overlaps with it
            # for each anchor, the max iou of all gts
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)  #找出bbox与所有GT的最大IOU[1]和对应GT的索引[3]
            # for each gt, which anchor best overlaps with it
            # for each gt, the max iou of all proposals
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)  #找出GT与bbox的最大IOU[0,0,0,1]和对应bbox的索引[0,0,0,0]

            # 2. assign negative: below
            # the negative inds are set to be 0,计算出来的max_iou>0但是比neg_thr=0.9小的proposal赋值0
            if isinstance(self.neg_iou_thr, float):  #0.9
                assigned_gt_inds[(max_overlaps >= 0)
                                 & (max_overlaps < self.neg_iou_thr)] = 0
            elif isinstance(self.neg_iou_thr, tuple):
                assert len(self.neg_iou_thr) == 2
                assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                                 & (max_overlaps < self.neg_iou_thr[1])] = 0

            # 3. assign positive: above positive IoU threshold
            pos_inds = max_overlaps >= self.pos_iou_thr   #计算出来的max_overlaps比pos=0.9大，就看成正例。
            #重点关注这里的+1操作，我觉得问题就出在这里....
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds]+1

            if self.match_low_quality:
                # Low-quality matching will overwrite the assigned_gt_inds
                # assigned in Step 3. Thus, the assigned gt might not be the
                # best one for prediction.
                # For example, if bbox A has 0.9 and 0.8 iou with GT bbox
                # 1 & 2, bbox 1 will be assigned as the best target for bbox A
                # in step 3. However, if GT bbox 2's gt_argmax_overlaps = A,
                # bbox A's assigned_gt_inds will be overwritten to be bbox B.
                # This might be the reason that it is not used in ROI Heads.
                for i in range(num_gts):
                    if gt_max_overlaps[i] >= self.min_pos_iou:
                        if self.gt_max_assign_all:
                            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                            assigned_gt_inds[max_iou_inds] = i + 1
                        else:
                            assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

            if gt_labels is not None: #这里考虑到当前是多标签数据集
                # 注意debug看一下这里的gt_labels是什么。我认为是表示gt_labels的个数，多标签必须大于1 .所以做了一个断言
                assert len(gt_labels[0]) > 1
                assigned_labels = assigned_gt_inds.new_zeros(
                    (num_bboxes, len(gt_labels[0])), dtype=torch.float32)

                # If not assigned, labels will be all 0
                pos_inds = torch.nonzero(
                    assigned_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_labels[
                        assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None

            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)   #我觉得这个分配的标签往后挪了一个，，，但是又不像。。。。。

else:
    # define an empty class, so that can be imported
    @import_module_error_class('mmdet')
    class MaxIoUAssignerAVA:
        pass
