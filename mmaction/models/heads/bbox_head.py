# Copyright (c) OpenMMLab. All rights reserved.
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.core.bbox import bbox_target
from mmaction.models.backbones.resnet3d_GCN import *
try:
    from mmdet.models.builder import HEADS as MMDET_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class BBoxHeadAVA(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        focal_alpha (float): The hyper-parameter alpha for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 1.
        focal_gamma (float): The hyper-parameter gamma for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 0.
        num_classes (int): The number of classes. Default: 81.
        dropout_ratio (float): A float in [0, 1], indicates the dropout_ratio.
            Default: 0.
        dropout_before_pool (bool): Dropout Feature before spatial temporal
            pooling. Default: True.
        topk (int or tuple[int]): Parameter for evaluating multilabel accuracy.
            Default: (3, 5)
        multilabel (bool): Whether used for a multilabel task. Default: True.
            (Only support multilabel == True now).
    """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2304,
            # The first class is reserved, to classify bbox as pos / neg
            focal_gamma=0.,
            focal_alpha=1.,
            num_classes=81,  #这里已经包含了80+1
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(3, 5),
            multilabel=True):

        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        #########################################################################################################################
        #这里读取一下类别占比的文件。在不同类别的loss前面乘一个权重系数self.clss_weight。
        # with open('/home/hjj/wuyini_pro/dataset/TITAN/annotations/categories.json', 'r') as fb:
        #     self.class_ratio = json.load(fb)
        # self.class_weight = torch.zeros(31)  #这里是对ratio中的32个类别。
        # self._init_class_weight()
        ##########################################################################################################################
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha


        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')
        # Class 0 is ignored when calculaing multilabel accuracy,
        # so topk cannot be equal to num_classes
        assert all([k < num_classes for k in self.topk])

        # Handle AVA first
        assert self.multilabel

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        #这里使用一个线性层，接入一个2304,输出一个num_classes=32.做分类
        self.fc_cls = nn.Linear(in_channels, num_classes)   #这里用一个线性层做了一个分类。
        self.debug_imgs = None

    ###########################################################################
    def _init_class_weight(self):
        for i in range(1, 32):  # 因为class_weight 0-31
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]
    ###########################################################################
    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):  #这里传进来的就是[27,2304,1,8,8].这里输入进来的就是增强过后的结合了短期特征的fbo特征。
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.temporal_pool(x)
        x = self.spatial_pool(x)
        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)  #按照size(0)=27进行一个reshape操作  [27,2304]
        cls_score = self.fc_cls(x)  #线性层就是[in_channel=2304, out_channel=类别数]  [27,32]
        # We do not predict bbox, so return None   这里不做bbox的。只返回一个分类分数
        return cls_score, None

    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    @staticmethod
    def recall_prec(pred_vec, target_vec):  #这里筛选出来的就是那些比thr=0.5高的检测值，以及大于写死的0.5的gt标签值
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec   #两个都为true，才判定为correct
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / target_vec.sum(1).float()
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()  #这里的cls_score就是pred.这一步将cls_score映射到[0,1]之间
        pred_vec = pred > thr  #这里选择的是比0.5大的数，返回的是形状为[27,32]的一堆false,true形式的值.我认为这里的thr可以根据我的实际情况做出调整。
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5 #这里的target选择的也是比0.5多的
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)  #这里获取的是根据阈值的出来的阈值recall和阈值prec

        recalls, precs = [], []
        for k in self.topk:  #(3,5)  k=3，5.对于3来说
            _, pred_label = pred.topk(k, 1, True, True) #从sigmoid后的cls_score中用topk，返回的是对应的值和索引
            pred_vec = pred.new_full(pred.size(), 0, dtype=torch.bool)

            num_sample = pred.shape[0]
            for i in range(num_sample):
                pred_vec[i, pred_label[i]] = 1
            recall_k, prec_k = self.recall_prec(pred_vec, target_vec)
            recalls.append(recall_k)
            precs.append(prec_k)
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            labels = labels[:, 1:]  #这里是取了这一批里面所有人从1开始的标签。它自动忽略了0这个类别。(3,31)
            pos_inds = torch.sum(labels, dim=-1) > 0  #感觉不到这一步有什么用。。。。
            cls_score = cls_score[pos_inds, 1:]   #只用了1-31的分类分数，没有管id=0
            labels = labels[pos_inds] #一行代表一个人，形状为(3,31)，31列里面出现哪个id，就是1，其他就是0

            bce_loss = F.binary_cross_entropy_with_logits  #该损失内部自带了计算logit的操作，无需在传入给这个损失函数之前手动使用sigmoid或者softmax函数将输入的cls_score映射到01之间
            logp = bce_loss(cls_score, labels, reduction='none')  #(3,31)的正值。reduction是对输出的结果做操作，None就是不做，默认mean取均值。这里还可以传进去一个weight参数，我觉得是可以做修改的。
            pt = torch.exp(-logp)  #e的-logp次方  #在cuda1上
            #############################################################
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * logp
            # 上面是原始的focal loss，我这里想给他乘一个根据ratio文件获取的权重
            # 在Focal loss后面乘一个。下面这里要指定一下device，不然不在一个地方。。。
            # F_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * logp * (self.class_weight.to(device="cuda:1"))   #class_weight在cpu上
            losses['loss_action_cls'] = torch.mean(F_loss)  #mean之后就是focal loss了。
            ###########################################################################
            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses
    ################################################################
    #这部分是我自己定义的多部分损失函数，年龄部分用一个损失，行为部分沿用原来的损失
    def loss_multi_part(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            labels = labels[:, 1:]  #这里是取了这一批里面所有人从1开始的标签。它自动忽略了0这个类别。(3,31)
            pos_inds = torch.sum(labels, dim=-1) > 0  #感觉不到这一步有什么用。。。。
            cls_score = cls_score[pos_inds, 1:]   #只用了1-31的分类分数，没有管id=0
            labels = labels[pos_inds] #一行代表一个人，形状为(3,31)，31列里面出现哪个id，就是1，其他就是0
            #方便起见共用一个bce_loss
            bce_loss = F.binary_cross_entropy_with_logits  #该损失内部自带了计算logit的操作，无需在传入给这个损失函数之前手动使用sigmoid或者softmax函数将输入的cls_score映射到01之间
            #for age group 年龄组的分类是一个多分类任务，从3个标签中选择一个即可。但是普通的多分类是一个数和三个数。这里的其实形状是一样的。
            age_score = cls_score[:,0:3]
            age_label = labels[:,0:3]
            logp_age = bce_loss(age_score,age_label,reduction='none')
            pt_age = torch.exp(-logp_age)
            age_F_loss = self.focal_alpha * (1 - pt_age)**self.focal_gamma * logp_age
            #for action group
            action_score = cls_score[:,3:]
            action_label = labels[:,3:]
            logp = bce_loss(action_score, action_label, reduction='none')  #(3,31)的正值。reduction是对输出的结果做操作，None就是不做，默认mean取均值。这里还可以传进去一个weight参数，我觉得是可以做修改的。
            pt = torch.exp(-logp)  #e的-logp次方
            action_F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * logp
            losses['loss_multi_part']=torch.mean(action_F_loss) + torch.mean(age_F_loss)  #组合一下新的loss形式。

            recall_thr_action, prec_thr_action, recall_k_action, prec_k_action = self.multi_label_accuracy(
                action_score, action_label, thr=0.5)
            losses['recall_action@thr=0.5'] = recall_thr_action
            losses['prec_action@thr=0.5'] = prec_thr_action
            for i, k in enumerate(self.topk):
                losses[f'recall_action@top{k}'] = recall_k_action[i]
                losses[f'prec_action@top{k}'] = prec_k_action[i]
        return losses
    ################################################################

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        assert self.multilabel

        scores = cls_score.sigmoid() if cls_score is not None else None
        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores


if mmdet_imported:
    MMDET_HEADS.register_module()(BBoxHeadAVA)
