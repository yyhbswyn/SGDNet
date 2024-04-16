# Copyright (c) OpenMMLab. All rights reserved.
import json
import pickle

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from mmaction.core.bbox import bbox_target

try:
    from mmdet.models.builder import HEADS as MMDET_HEADS

    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = np.nan_to_num(_adj, nan=-1)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_adj(A):
    A = torch.tensor(A)
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    A = A.float()
    middle = torch.matmul(A, D).t()
    adj = torch.matmul(middle, D)
    return adj



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        weight = self.weight
        weight = weight.to(device)
        input = input.float()
        input = input.to(device)
        support = torch.matmul(input, weight)
        # print("input：",input.shape)   #32,2304
        # print("weight：",weight.shape)  #[2304,2304]
        # print("support：",support.shape)  #[32,2304]
        support = support.to(device)
        adj = adj.to(device)
        output = torch.matmul(adj, support)  #[32,2304]
        # print("output：",output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBoxHeadTITAN(nn.Module):
    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2304,
            focal_gamma=0.,
            focal_alpha=1.,
            num_classes=81,
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(3, 5),
            multilabel=True):

        super(BBoxHeadTITAN, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        ##For GCN inference
        self.gc1_word = GraphConvolution(300, 2304)
        self.gc1_sentence = GraphConvolution(384, 2304)
        self.gc2 = GraphConvolution(2304, 2304)
        self.gcn_relu = nn.LeakyReLU(0.2)
        self.A = gen_A(num_classes=32, t=0.5, adj_file='your_path/new_adj.pkl')  #[0]是共现矩阵，32*32，1是出现次数(32,)
        self.adj = gen_adj(self.A).detach()
        #true pkl
        word_file = open("your_path/32wordembedding.pkl", "rb")
        sentence_file = open("your_path/sentence_embedding411.pkl", "rb")
        self.word_embedding = torch.from_numpy(pickle.load(word_file))
        self.sentence_embedding = torch.from_numpy(pickle.load(sentence_file))

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk,)
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
        # linear layer
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)

        nn.init.constant_(self.fc_cls.bias, 0)


    def forward(self, bbox):
        if self.dropout_before_pool and self.dropout_ratio > 0:
            bbox = self.dropout(bbox)
        bbox = self.temporal_pool(bbox)
        bbox = self.spatial_pool(bbox)
        if not self.dropout_before_pool and self.dropout_ratio > 0:
            bbox = self.dropout(bbox)
        bbox = bbox.view(bbox.size(0), -1)

        gcn_x = self.gc1_sentence(self.sentence_embedding, self.adj)

        gcn_x = self.gcn_relu(gcn_x)
        gcn_x = self.gc2(gcn_x, self.adj)
        gcn_x = gcn_x.transpose(0, 1)
        cls_score = torch.matmul(bbox, gcn_x) #[27,32]

        return cls_score, None



    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]  # Groundtruth classification label list.

        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)

        return cls_reg_targets

    @staticmethod
    def recall_prec(pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / target_vec.sum(1).float()
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

        recalls, precs = [], []
        for k in self.topk:
            _, pred_label = pred.topk(k, 1, True, True)
            pred_vec = pred.new_full(pred.size(), 0, dtype=torch.bool)

            num_sample = pred.shape[0]
            for i in range(num_sample):
                pred_vec[i, pred_label[i]] = 1
            recall_k, prec_k = self.recall_prec(pred_vec, target_vec)
            recalls.append(recall_k)
            precs.append(prec_k)
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,  # 这里传进入的是cls_score，应该是我用GCN计算出来的东西[number,32]
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score  #
            labels = labels[:, 1:]
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds, 1:]
            labels = labels[pos_inds]

            bce_loss = F.binary_cross_entropy_with_logits

            loss = bce_loss(cls_score, labels, reduction='none')
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * loss
            losses['loss_action_cls'] = torch.mean(F_loss)

            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses

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
    MMDET_HEADS.register_module()(BBoxHeadTITAN)
