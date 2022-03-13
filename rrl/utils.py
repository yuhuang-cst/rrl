# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Yu Huang
# @Email: yuhuang-cst@foxmail.com

import os
import chardet
import numpy as np
import logging
import random
import string

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

def randstr(num):
    return ''.join(random.sample(string.ascii_letters + string.digits, num))


def get_logger(name, log_path=None, level=logging.DEBUG, mode='a'):
    """
    Args:
        name (str or None): None means return root logger
        log_path (str or None): log文件路径
    """
    formatter = logging.Formatter(fmt="%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        return logger
    logger.setLevel(level)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def delete_logger(logger):
    while logger.handlers:
        logger.handlers.pop()

def run_sys_cmd(cmd):
    r = os.popen(cmd)
    bytes = r.buffer.read()
    if bytes:
        coding = chardet.detect(bytes)['encoding']
        lines = bytes.decode(encoding=coding).splitlines()
        return [line.strip() for line in lines]
    return []


def get_gpu_memory_list():
    """
    Returns:
        list of int: MB
    """
    lines = run_sys_cmd('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    memory_gpu = [int(x.split()[2]) for x in lines]
    return memory_gpu


def get_one_idle_gpu():
    """
    Returns:
        str: seleted gpu
        int: memory (MB)
    """
    memory_gpu = get_gpu_memory_list()
    if memory_gpu:
        ret_gpu = int(np.argmax(memory_gpu))
        return str(ret_gpu), memory_gpu[ret_gpu]
    return None, None


def get_idle_gpus(max_num=None):
    """
    Args:
        max_num:
    Returns:
        list: list of gpu_str
        list: list of memory of corresponding cpu
    """
    memory_gpu = get_gpu_memory_list()
    if max_num is None:
        max_num = len(memory_gpu)
    else:
        max_num = min(max_num, len(memory_gpu))
    gpu_list = np.argsort(memory_gpu)[-max_num:][::-1]
    return [str(gpu_id) for gpu_id in gpu_list], [memory_gpu[gpu_id] for gpu_id in gpu_list]


def get_auc_dict(y_true, y_score_mat, label_list):
    """
    Args:
        y_true: (n_samples,)
        y_score_mat: (n_samples, n_classes)
        label_list (list): List of labels that index the classes in ``y_score``
    Returns:
        dict: recall_dict
        dict: precision_dict
    """
    lb_to_rank = {lb: i for i, lb in enumerate(label_list)}
    fpr_dict, tpr_dict = {}, {}
    for lb in label_list:
        y_score = y_score_mat[:, lb_to_rank[lb]]
        fpr_dict[lb], tpr_dict[lb], _ = roc_curve(y_true, y_score, pos_label=lb)
    return fpr_dict, tpr_dict


def get_cls_metric_dict(y_true, y_pred, y_score=None, label_list=None, cal_auc=False, cal_confusion=False):
    """
    Args:
        y_true (array-like): (n_samples,)
        y_pred (array-like): (n_samples,)
        y_score (array-like): (n_samples, n_classes)
        label_list (array-like): column names of y_score
        cal_auc (bool): calculate AUC
        cal_confusion: calculate confusion matrix
    Returns:
        dict: {
            label_1: {
                'precision': float,
                'recall': float,
                'f1-score': float,
                'support': int
            },
            'accuracy': float
            'macro avg': {...},
            'micro avg': {...},
            'weighted avg': {...},
            ...
        }
    """
    label_list = np.array(label_list)
    d = classification_report(list(y_true), list(y_pred), output_dict=True)
    if cal_confusion:
        assert label_list is not None
        d['confusion matrix'] = confusion_matrix(list(y_true), list(y_pred), labels=label_list).tolist()
    if cal_auc:
        assert y_score is not None and label_list is not None
        fpr_dict, tpr_dict = get_auc_dict(y_true, y_score, label_list)
        for lb in label_list:
            d[lb]['auc'] = auc(fpr_dict[lb], tpr_dict[lb])
        if len(np.unique(y_true)) == 2 and len(y_score.shape) == 2:
            y_score = y_score[:, 1]
        d['macro avg']['auc'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr', labels=label_list)  # default
        if 'micro avg' not in d:
            d['micro avg'] = {}
        d['micro avg']['auc'] = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr', labels=label_list)
        d['weighted avg']['auc'] = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr', labels=label_list)
    return d


def flatten_dict(d, prefix='', sep='-'):
    if not isinstance(d, dict):
        return {prefix: d}
    ret = {}
    for k, v in d.items():
        p = prefix + sep + k if prefix else k
        ret.update(flatten_dict(v, p, sep))
    return ret


if __name__ == '__main__':
    gpus, _ = get_idle_gpus()
    print(gpus)
    pass

