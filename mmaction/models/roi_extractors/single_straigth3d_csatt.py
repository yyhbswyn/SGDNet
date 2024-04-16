# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.utils import import_module_error_class

try:
    from mmcv.ops import RoIAlign, RoIPool
except (ImportError, ModuleNotFoundError):

    @import_module_error_class('mmcv-full')
    class RoIAlign(nn.Module):
        pass

    @import_module_error_class('mmcv-full')
    class RoIPool(nn.Module):
        pass


try:
    from mmdet.models import ROI_EXTRACTORS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class ChannelAtt(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAtt, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAtt(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAtt, self).__init__()
        #声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  #平均池化[5,1,8,8]
        max_out, _ = torch.max(x, dim=1, keepdim=True) #最大池化[5,1,8,8]
        #拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) #7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


class CSATT(nn.Module):
    def __init__(self):
        super(CSATT, self).__init__()
        self.catt=ChannelAtt(2304)
        self.satt=SpatialAtt()
        self.infactor = nn.Parameter(torch.zeros(1))  #给一个可学习的参数。用来控制经过注意力的F和原始F的相加情况

    def forward(self,x):
        #先分别获取两种注意力映射，通道注意力映射c,空间注意力映射s
        c = self.catt(x)  #[5,2304,1,1]
        s = self.satt(x)  #[5,1,8,8]
        f1 = torch.mul(x,c) #[5,2304,8,8]
        f2 = torch.mul(f1,s)  #[5,2304,8,8]
        out = self.infactor*f2+x
        return out




class SingleRoIExtractor3D_CSATT(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
        featmap_stride (int): Strides of input feature maps. Default: 16.
        output_size (int | tuple): Size or (Height, Width). Default: 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Default: 0.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
            Default: 'avg'.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Default: True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Default: False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.

    这里的ROIAlign就是输入一个feature map。对于每个不同尺寸的proposal region，转换为固定大小的H*W的feature map，H和W是超参数
    比如，对于一个7*5大小的region,可以把它按照H*W=2*2去划分成4个长方形格子。对每个格子里面也分4个区，4个区的数值通过双线性插值做一个融合。一个格子最终融合成4个区值。
    一个区域里一共4个格子，池化后有4个采样图。这样拼起来的4个采样图就是该区域固定大小的输出。连接全连接层进行回归和分类。
    这里的双线性插值，对于一个区内的一个点的插值，就是它附近4个点的加权和。权值对应的就是它在总面积中的占比。
    """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 temporal_pool_mode='avg',
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global


        """这里是我定义的channel spatial注意力模块"""
        self.csatt=CSATT()



        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(    #我选择的是RoIAlign
                self.output_size,    #我写的8
                self.spatial_scale,  #1/sample_stride
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois):
        # slowfast模型，它输入进来的feat是一个tuple类型的Out(x_slow,x_fast)
        if not isinstance(feat, tuple):   #用来判断这个feat是不是我们指定的tuple类型。这里的tuple其实就是out(slow,fast)
            feat = (feat, )

        if len(feat) >= 2:
            # print("对于输入的feat进行roi特征提取，如果len(feat)>=2")
            maxT = max([x.shape[2] for x in feat])   #MaxT就是从这个元组的数据里面找。找slow和fast的时间最长的那个数据
            max_shape = (maxT, ) + feat[0].shape[3:]   #获取maxT=64,后面拼上去feat[0].shape[3:]  就是第一组数据只要后面的WH   [64,16,16]
            # resize each feat to the largest shape (w. nearest)   #把每个元组feat都做个形状变化，变到[5,channels,64,16,16]
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]    #对[5,channels,T,16,16]的第[2]维T做均值。化成了[5,channel,1,16,16]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()   #这里把元组按照第[1]的channel维度进行了cat操作，[5,2304,1,16,16]

        """这里通过我修改的通道和空间注意力，这里是通道和空间注意力模块。"""
        feat = torch.squeeze(feat, dim=2)
        feat = self.csatt(feat)   #这里是结合了原始特征和经过注意力特征的输出。
        feat = torch.unsqueeze(feat,dim=2)
        """上面是做了一个通道和空间注意力模块Channel spatial"""



        roi_feats = []
        for t in range(feat.size(2)):   #feat.size(2)就是指[5,2304,1,16,16]中的第[2]个维度1  t=0,1
            frame_feat = feat[:, :, t].contiguous()   #feat[:,:,0]=[5,2304,16,16] 这里把时间这个去掉了
            roi_feat = self.roi_layer(frame_feat, rois)   #[5,2304,16,16], [27,5] 这两个数通过roi_layer之后变成了【27,2304,8,8】  因为我设置的outputsize是8
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        return torch.stack(roi_feats, dim=2), feat  #这里用stack函数，把T维度加上。roi_feats=[number,2304,1,8,8]  feat为[5,2304,1,16,16]


if mmdet_imported:
    ROI_EXTRACTORS.register_module()(SingleRoIExtractor3D_CSATT)
