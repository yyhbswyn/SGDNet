# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox2result(bboxes, labels, num_classes, thr=0.01):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 4)
        labels (Tensor): shape (n, #num_classes)
        num_classes (int): class number, including background class
        thr (float): The score threshold used when converting predictions to
            detection results
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))

    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    # We only handle multilabel now
    assert labels.shape[-1] > 1

    scores = labels  # rename for clarification
    thr = (thr, ) * num_classes if isinstance(thr, float) else thr   #thr=0.002,长度是32,每个都是0.002
    assert len(thr) == num_classes   #len(thr)=32

    result = []
    for i in range(num_classes - 1):   #i=0````31.一共有32个数字,但是score只有31个数据
        where = scores[:, i + 1] > thr[i + 1]  #i=1,32.而i只有可能在0-30才有数据.i=0就是第一个score,依次往后.0-30是所有的数据.我怀疑因为这里的thr只有0-31个索引,而
        result.append(
            np.concatenate((bboxes[where, :4], scores[where, i + 1:i + 2]),
                           axis=1))
    return result
