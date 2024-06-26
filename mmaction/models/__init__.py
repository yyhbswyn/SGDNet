# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import (C3D, STGCN, X3D, MobileNetV2, MobileNetV2TSM, ResNet,
                        ResNet2Plus1d, ResNet3d, ResNet3dCSN, ResNet3dLayer,
                        ResNet3dSlowFast, ResNet3dSlowOnly, ResNetAudio,
                        ResNetTIN, ResNetTSM, TANet, TimeSformer
                        ,YOWO_BACKBONE)
from .builder import (BACKBONES, DETECTORS, HEADS, LOCALIZERS, LOSSES, NECKS,
                      RECOGNIZERS, build_backbone, build_detector, build_head,
                      build_localizer, build_loss, build_model, build_neck,
                      build_recognizer)
from .common import (LFB, TAM, Conv2plus1d, ConvAudio,
                     DividedSpatialAttentionWithNorm,
                     DividedTemporalAttentionWithNorm, FFNWithNorm)
from .heads import (ACRNHead, AudioTSNHead, AVARoIHead,TITANRoIHead, BaseHead, BBoxHeadAVA,
                    FBOHead, I3DHead, LFBInferHead, SlowFastHead, STGCNHead,
                    TimeSformerHead, TPNHead, TRNHead, TSMHead, TSNHead,BBoxHeadTITAN,BBoxHeadTITAN_Learn,
                    X3DHead,
                    ACARRoIHead,ACARHeadAVA,  #自己复现的两个ACAR head
                    YOLOV3_HEAD)
from .localizers import BMN, PEM, TEM
from .losses import (BCELossWithLogits, BinaryLogisticRegressionLoss, BMNLoss,
                     CrossEntropyLoss, HVULoss, NLLLoss, OHEMHingeLoss,
                     SSNLoss)
from .necks import TPN
from .recognizers import (AudioRecognizer, BaseRecognizer, Recognizer2D,
                          Recognizer3D)
from .roi_extractors import SingleRoIExtractor3D, SingleRoIExtractor3D_ContextRcnn_csatt,SingleRoIExtractor3D_ContextRcnn
from .skeleton_gcn import BaseGCN, SkeletonGCN



__all__ = [
    'BACKBONES', 'HEADS', 'RECOGNIZERS', 'build_recognizer', 'build_head',
    'build_backbone', 'Recognizer2D', 'Recognizer3D', 'C3D', 'ResNet', 'STGCN',
    'ResNet3d', 'ResNet2Plus1d', 'I3DHead', 'TSNHead', 'TSMHead', 'BaseHead',
    'STGCNHead', 'BaseRecognizer', 'LOSSES', 'CrossEntropyLoss', 'NLLLoss',
    'HVULoss', 'ResNetTSM', 'ResNet3dSlowFast', 'SlowFastHead', 'Conv2plus1d',
    'ResNet3dSlowOnly', 'BCELossWithLogits', 'LOCALIZERS', 'build_localizer',
    'PEM', 'TAM', 'TEM', 'BinaryLogisticRegressionLoss', 'BMN', 'BMNLoss',
    'build_model', 'OHEMHingeLoss', 'SSNLoss', 'ResNet3dCSN', 'ResNetTIN',
    'TPN', 'TPNHead', 'build_loss', 'build_neck', 'AudioRecognizer',
    'AudioTSNHead', 'X3D', 'X3DHead', 'ResNet3dLayer', 'DETECTORS',
    'SingleRoIExtractor3D', 'BBoxHeadAVA', 'BBoxHeadTITAN','ResNetAudio', 'build_detector',
    'ConvAudio', 'AVARoIHead', 'MobileNetV2', 'MobileNetV2TSM', 'TANet', 'LFB',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'NECKS', 'TimeSformer',
    'TimeSformerHead', 'DividedSpatialAttentionWithNorm',
    'DividedTemporalAttentionWithNorm', 'FFNWithNorm', 'ACRNHead', 'BaseGCN',
    'SkeletonGCN',
    'YOWO_BACKBONE',
    'YOLOV3_HEAD',
    #自己复现的两个
    'ACARRoIHead',
    'ACARHeadAVA',
    'SingleRoIExtractor3D_ContextRcnn_csatt',
    'SingleRoIExtractor3D_ContextRcnn'

]
