# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS

import pickle
@DATASETS.register_module()
class AVADataset(BaseDataset):
    """AVA dataset for spatial temporal detection.

    Based on official AVA annotation files, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Default: None.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        custom_classes (list[int]): A subset of class ids from origin dataset.
            Please note that 0 should NOT be selected, and ``num_classes``
            should be equal to ``len(custom_classes) + 1``
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website. Default: 902.
        timestamp_end (int): The end point of included timestamps. The
            default value is referred from the official website. Default: 1798.
    """

    _FPS = 30

    def __init__(self,
                 ann_file,
                 exclude_file,
                 pipeline,
                 label_file=None,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=81,
                 custom_classes=None,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=2000,
                 timestamp_start=2,

                 ):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = tuple([0] + custom_classes)
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.num_classes = num_classes
        self.filename_tmpl = filename_tmpl
        #在这里指定一下这两个mode
        self.test_mode = test_mode

        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        # self.timestamp_end = timestamp_end
        self.logger = get_root_logger()


        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,

            start_index=start_index,
            modality=modality,
            num_classes=num_classes)

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None

        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.logger.info(
                f'{len(valid_indexes)} out of {len(self.video_infos)} '
                f'frames are valid.')
            self.video_infos = [self.video_infos[i] for i in valid_indexes]



    def parse_img_record(self, img_records):
        """Merge image records of the same entity at the same time.
        合并同一实体的image记录。
        Args:
            img_records (list[dict]): List of img_records (lines in AVA
                annotations).这里一次传进来的是同一个视频的同一帧中的所有数据。
                比如。这里传进来的img_records长度为28.因为注释文件中的331视频的第2帧一共有28条数据。

        Returns:
            tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids
        """
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            num_selected_records = len(selected_records)
            img_records = [
                x for x in img_records if
                not np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            assert len(img_records) + num_selected_records == num_img_records   #26+2=28

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def filter_exclude_file(self):
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.video_infos)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.video_infos):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes
#########################################################################################################
    def get_len_count(self):
        #titan
        txt_infos = []
        #for test set
        if self.test_mode is True:
            file = open('your_data_path/TITAN/annotations/test_count.txt', "r")
            for index in file:
                txt_info_test = index.strip('\n')
                txt_infos.append(txt_info_test)
            file.close()
        #for train set
        else:
            file = open('/data3/wuyini_dataset/dataset/TITAN/annotations/train_val_count.txt', "r")
            for index in file:
                txt_info_test = index.strip('\n')
                txt_infos.append(txt_info_test)
            file.close()
        return txt_infos

        ###for lfb code
        # txt_infos = []
        # file = open('/data3/wuyini_dataset/dataset/TITAN/annotations/lfb_count.txt', "r")
        # for index in file:
        #     txt_info_test = index.strip('\n')
        #     txt_infos.append(txt_info_test)
        # file.close()
        # return txt_infos


    def get_shot_info(self,current_video_id,len_count_infos):
        shot=()
        for len_info in len_count_infos:
            name = len_info.split(' ')[0]
            video_len = float(len_info.split(' ')[1])
            if (current_video_id == name):
                shot = (0, video_len * self._FPS)
            else:
                continue

        return shot
#################################################################################################################
    def load_annotations(self):
        """Load AVA annotations."""
        video_infos = []
        records_dict_by_img = defaultdict(list)
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')

                label = int(line_split[6])  #读取行为标签
                if self.custom_classes is not None:
                    if label not in self.custom_classes:
                        continue
                    label = self.custom_classes.index(label)

                video_id = line_split[0]   #视频id号
                timestamp = int(line_split[1])  #第几帧
                img_key = f'{video_id},{timestamp:04d}'

                entity_box = np.array(list(map(float, line_split[2:6])))   #人的坐标
                entity_id = int(line_split[7])  #人的id

                # shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                #              self._FPS)
                txt_infos = self.get_len_count()
                shot_info = self.get_shot_info(video_id,txt_infos)
                video_info = dict(
                    video_id=video_id,
                    timestamp=timestamp,
                    entity_box=entity_box,
                    label=label,
                    entity_id=entity_id,
                    shot_info=shot_info)
                records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann)
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index   #1
        results['timestamp_start'] = self.timestamp_start   #2
        # results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']
        return self.pipeline(results)


    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index   #1
        results['timestamp_start'] = self.timestamp_start   #2
        # results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        # Follow the mmdet variable naming style.
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)

    def dump_results(self, results, out):
        """Dump predictions into a csv file."""
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)

    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        """Evaluate the prediction results and report mAP."""
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        results2csv(self, results, temp_file, self.custom_classes)

        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            eval_result = ava_eval(
                temp_file,
                metric,
                self.label_file,
                self.ann_file,
                self.exclude_file,
                custom_classes=self.custom_classes)
            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        os.remove(temp_file)

        return ret
