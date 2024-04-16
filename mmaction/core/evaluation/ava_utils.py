# Copyright (c) OpenMMLab. All rights reserved.
import csv
import logging
import time
from collections import defaultdict

import numpy as np

from .ava_evaluation import object_detection_evaluation as det_eval
from .ava_evaluation import standard_fields


def det2csv(dataset, results, custom_classes): #dataset就是读取到的video_infos,results中是检测到的Bbox信息 673
    print("2.这里ava_utils中的det2csv方法")
    csv_results = []
    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx]['video_id']
        timestamp = dataset.video_infos[idx]['timestamp']
        result = results[idx] #每一个result中存放的是一个人的每一个行为类别的置信度值
        for label, _ in enumerate(result):  #对每一个result读取对应的label信息
            for bbox in result[label]:
                bbox_ = tuple(bbox.tolist())
                if custom_classes is not None:
                    actual_label = custom_classes[label + 1]
                else:
                    actual_label = label + 1  #这里就是获取真正的行为类别id
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:4] + (actual_label, ) + bbox_[4:])  #获得视频、帧、坐标、行为标签从1开始的、置信度
    return csv_results


# results is organized by class
def results2csv(dataset, results, out_file, custom_classes=None):
    print("1.这里是ava_utils中的results2csv方法")
    if isinstance(results[0], list):
        csv_results = det2csv(dataset, results, custom_classes)

    # save space for float
    def to_str(item):
        if isinstance(item, float):
            return f'{item:.3f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(to_str, csv_result)))
            f.write('\n')


def print_time(message, start):
    print('==> %g seconds to %s' % (time.time() - start, message), flush=True)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv(csv_file, class_whitelist=None):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class
        labels not in this set are skipped.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list
        of integer class labels, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list
        of score values labels, matching the corresponding label in `labels`.
        If scores are not provided in the csv, then they will default to 1.0.
    """
    print("5.这里ava_utils中的read_csv方法")
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], 'Wrong number of columns: ' + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 8:  #这里其实只对后面读取result.csv中有用。因为detection result中的最后一列才是行为的置信度值。而如果是在读test.csv的话，会把score变成人的id。
            score = float(row[7])   #这里很奇怪。。。。为什么要命名为score.这里row[7]是我的人的id啊。。。。

        entries[image_key].append((score, action_id, y1, x1, y2, x2))  #score就变成0了。。。好无语啊。。。。。接着看一下这个score会在哪里用到

    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        boxes[image_key] = [x[2:] for x in entry]
        labels[image_key] = [x[1] for x in entry]
        scores[image_key] = [x[0] for x in entry]

    print_time('read file ' + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    print("6.这里ava_utils中的read_exclusions方法")
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, 'Expected only 2 columns, got: ' + row
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    print("4.这里ava_utils中的read_labelmap方法")
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


# Seems there is at most 100 detections for each image
def ava_eval(result_file,
             result_type,
             label_file,
             ann_file,
             exclude_file,
             verbose=True,
             custom_classes=None):

    assert result_type in ['mAP']
    print("3.这里ava_utils中的ava_eval方法")

    start = time.time()
    categories, class_whitelist = read_labelmap(open(label_file))
    if custom_classes is not None:
        custom_classes = custom_classes[1:]
        assert set(custom_classes).issubset(set(class_whitelist))
        class_whitelist = custom_classes
        categories = [cat for cat in categories if cat['id'] in custom_classes]

    # loading gt, do not need gt score。这里不影响，就算读了类别也无所谓。。因为根本没用到最后一列值。。。。所以用了一个_来代替
    gt_boxes, gt_labels, _ = read_csv(open(ann_file), class_whitelist)
    if verbose:
        print_time('Reading detection results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    boxes, labels, scores = read_csv(open(result_file), class_whitelist)   #感觉这里的score就是，一帧里面有n个人，就会有n*31长度的分数。
    if verbose:
        print_time('Reading detection results', start)

    # Evaluation for mAP。定义了一个检测器叫做pascal_evaluator
    pascal_evaluator = det_eval.PascalDetectionEvaluator(categories)  #然后调用检测方法。这里的iou是0.5。到时候需要在配置文件里面看一看这个0.5是不是我自己定的。

    start = time.time()
    for image_key in gt_boxes:
        if verbose and image_key in excluded_keys:
            logging.info(
                'Found excluded timestamp in detections: %s.'
                'It will be ignored.', image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(  #这里传进去单张图的bbox和classes
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array(gt_boxes[image_key], dtype=float),  #一个人在这一帧有几个动作，就会有几个完全一样的坐标框 list就等于几。这一步就是转成numpy（n,4）
                standard_fields.InputDataFields.groundtruth_classes:
                np.array(gt_labels[image_key], dtype=int)
            })
    if verbose:
        print_time('Convert groundtruth', start)

    start = time.time()
    for image_key in boxes:
        if verbose and image_key in excluded_keys:
            logging.info(
                'Found excluded timestamp in detections: %s.'
                'It will be ignored.', image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                np.array(scores[image_key], dtype=float)
            })
    if verbose:
        print_time('convert detections', start)

    start = time.time()
    metrics = pascal_evaluator.evaluate()   #传进去之后，调用这个评估方法进行度量。
    # age_metrics,base_metrics,traffic_metrics,non_traffic_metrics = pascal_evaluator.evaluate()   #传进去之后，调用这个评估方法进行度量。
    if verbose:
        print_time('run_evaluator', start)
    for display_name in metrics:
        print(f'this is the eval result:  {display_name}=\t{metrics[display_name]}')
    # for age_display_name in age_metrics:
    #     print(f'age-group:{age_display_name}=\t{age_metrics[age_display_name]}')
    # for base_display_name in base_metrics:
    #     print(f'base-action:{base_display_name}=\t{base_metrics[base_display_name]}')
    # for traffic_display_name in traffic_metrics:
    #     print(f'traffic-action:{traffic_display_name}=\t{traffic_metrics[traffic_display_name]}')
    # for non_traffic_display_name in non_traffic_metrics:
    #     print(f'non-traffic-action:{non_traffic_display_name}=\t{non_traffic_metrics[non_traffic_display_name]}')
    return {
        display_name: metrics[display_name]
        for display_name in metrics if 'ByCategory' not in display_name
    }
