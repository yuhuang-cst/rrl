# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reference: https://github.com/12wang3/rrl

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import scipy.sparse as sp
from loguru import logger
from tqdm import tqdm

from rrl.components import BinarizeLayer
from rrl.components import UnionLayer, LRLayer
from rrl.feature_order_manager import FeatureOrderManager
from rrl.utils import get_one_idle_gpu, get_cls_metric_dict, flatten_dict
from rrl.constant import AND_OP, OR_OP, NOT_OP


class MLLP(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, estimated_grad=False):
        super(MLLP, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 4:
                num += self.layer_list[-2].output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            else:
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        return self.continuous_forward(x), self.binarized_forward(x)

    def continuous_forward(self, x):
        x_res = None
        for i, layer in enumerate(self.layer_list):
            if i <= 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i <= 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list


class RRLClassifier(object):
    def __init__(self, dim_list=None, use_not=False, estimated_grad=False, continuous_left=-3., continuous_right=3.,
                 distributed=False, save_folder=None, save_mode='best', dtype=torch.float32, auto_val_size=0.05, device=None,
                 epoch=50, lr=0.001, lr_decay_epoch=None, lr_decay_rate=0.75, batch_size=32, weight_decay=0.0, log_step=50, eval_step=50,
                 pin_memory=False, write_summary=True, eval_metric='macro avg f1-score', verbose=True):
        """
        Args:
            dim_list (list): [n_hidden1, n_hidden2]; Note: the input layer (n_disc_features, n_conti_features) is not included. Default: [1, 16]
            use_not (bool): Use the NOT (~) operator in logical rules. It will enhance model capability but make the RRL more complex.
            estimated_grad (bool): Use estimated gradient
            continuous_left (float or array-like): min value for continuous features; deault: -3.0
            continuous_right (float or array-like): max value for continuous features; deault: 3.0
            distributed (bool): TODO: Use multiple gpu to train model.
            save_folder (str): save path of model. e.g. 'model.pth'
            save_mode (str or None): 'best' | 'end' | None
                'best': save the best model during training accrording to performance in valid dataset
                'end': save in the end of training
                None: do not save
            dtype (np.dtype): dtype of X, y
            auto_val_size (float): generate validation set if it is not provided.
            epoch (int): rounds of going through training set
            lr (float): initial learning rate
            lr_decay_epoch (int or None): learning rate decay epoch. default: None (not used lr decay)
            lr_decay_rate (float): learning rate decay rate
            batch_size (int)
            weight_decay (float): L2 penalty
            log_step (int): The number of batches to log once
            eval_step (int): The number of batches to evaluate once
            pin_memory (bool): whether to use pin_memory
            write_summary (bool): whether to use write summary which can be shown by tensorboard
            eval_metric (str): metric to select the best model during traning. e.g. 'macro avg f1-score', 'accuracy', 'weighted avg precision', '{class_name} f1-score'
        """
        super(RRLClassifier, self).__init__()
        self.dim_list = self.check_dim_list(dim_list or [1, 16])
        self.use_not = use_not
        self.estimated_grad = estimated_grad
        self.continuous_left = continuous_left
        self.continuous_right = continuous_right
        self.distributed = distributed
        self.save_mode = save_mode
        self.epoch = epoch
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_rate = lr_decay_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.log_step = log_step
        self.eval_step = eval_step
        self.dtype = dtype
        self.auto_val_size = auto_val_size
        self.device = self.get_default_device() if device is None else device
        self.pin_memory = pin_memory
        self.write_summary = write_summary
        self.eval_metric = eval_metric
        self.verbose = verbose

        self.init_save_path(save_folder)
        self.feat_order_manager = None


    def logger_info(self, *args, **kwargs):
        if self.verbose:
            logger.info(*args, **kwargs)


    def init_save_path(self, save_folder):
        self.save_folder = save_folder or './rrl_model'
        self.model_path = os.path.join(self.save_folder, 'model.pth')
        self.log_path = os.path.join(self.save_folder, 'log.txt')
        self.summary_folder = os.path.join(self.save_folder, 'summary')


    def get_default_device(self):
        if torch.cuda.is_available():
            gpu_id, gpu_mem = get_one_idle_gpu()
            print(f'Use GPU:{gpu_id}; GPU memory available = {gpu_mem} MB')
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cpu')


    def set_device(self, device):
        self.device = device


    def __del__(self):
        self.remove_logger()


    def check_dim_list(self, dim_list):
        assert len(dim_list) >= 2, 'length of dim_list should be at least 2'
        return dim_list


    def init_logger(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if hasattr(self, 'log_hander_id') and self.log_hander_id is not None:
            self.log_hander_id = logger.add(self.log_path)


    def remove_logger(self):
        if hasattr(self, 'log_hander_id') and self.log_hander_id is not None:
            logger.remove(self.log_hander_id)
            self.log_hander_id = None


    def convert_mat_to_dataloader(self, X, batch_size, shuffle, y=None):
        if sp.issparse(X):
            if not sp.isspmatrix_coo(X):
                X = X.tocoo()
            X = torch.sparse_coo_tensor(torch.LongTensor([X.row.tolist(), X.col.tolist()]), X.data, dtype=self.dtype)
        else:
            X = torch.tensor(X, dtype=self.dtype)
        if y is None:
            dataset = TensorDataset(X)
        else:
            y = torch.tensor(y, dtype=torch.int64)
            dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=self.pin_memory)


    def check_X(self, X, discrete_features=None):
        if discrete_features is not None:
            X_disc = X[:, discrete_features]
            assert (X_disc == 1).sum() + (X_disc == 0).sum() == X.shape[0] * X.shape[1]
        return X


    def fit(self, X, y, sample_wieght=None, discrete_features=None, X_val=None, y_val=None):
        """Fit using matrix
        Args:
            X (array-like): (n_samples, n_features); Note: discrete features should be one-hot
            y (array-like): (n_samples,)
            sample_wieght (array-like): (n_samples,) default=None TODO: implement sample_weight
            discrete_features (array-like): A one-dimensional array of categorical columns indices. default: None (all features are considered continuous)
            X_val (array-like or None): (n_samples_val, n_features)
            y_val (array-like or None): (n_samples_val,)
        Returns:
            RRL
        """
        self.init_logger()
        X = self.check_X(X)
        if discrete_features:
            self.feat_order_manager = FeatureOrderManager()
            self.feat_order_manager.fit_transform(X, discrete_features, inplace=True)
            input_dims = (len(discrete_features), X.shape[1] - len(discrete_features))
        else:
            input_dims = (0, X.shape[1])
            self.feat_order_manager = None

        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        self.classes_ = label_encoder.classes_
        self.n_classes = len(self.classes_)
        self.need_to_map_y = True
        if (self.classes_ == np.arange(self.n_classes, dtype=self.classes_.dtype)).all():
            self.need_to_map_y = False

        train_loader = self.convert_mat_to_dataloader(X, y=y, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self.convert_mat_to_dataloader(X_val, y=y_val, batch_size=self.batch_size, shuffle=False)

        self.fit_dataloader(input_dims, train_loader, val_loader=val_loader)
        self.remove_logger()
        if self.feat_order_manager is not None:
            self.feat_order_manager.inverse_transform(X, inplace=True)


    def init_network(self):
        net = MLLP(
            self.dim_list, use_not=self.use_not, left=self.continuous_left,
            right=self.continuous_right, estimated_grad=self.estimated_grad
        )
        # if self.distributed:
        #     net = MyDistributedDataParallel(self.net, device_ids=[self.device_id])
        return net


    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()


    def fit_dataloader(self, input_dims, train_loader, val_loader=None):
        """Fit using dataloader
        Args:
            input_dims (tuple): (disc_dim, continuous_dim)
            train_loader (torch.utils.data.DataLoader):
            val_loader (torch.utils.data.DataLoader):
            label_encoder (LabelEncoder)
        """
        self.init_logger()
        if val_loader is None and self.auto_val_size > 0.:
            val_size = int(len(train_loader.dataset) * self.auto_val_size)
            train_size = len(train_loader.dataset) - val_size
            train_ds, val_ds = random_split(train_loader.dataset, [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)

        self.dim_list = [input_dims] + (self.dim_list or []) + [self.n_classes]
        self.logger_info(f'Dim list: {self.dim_list}')
        self.net = self.init_network().to(self.device)
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sum_writer = None
        if self.write_summary:
            os.makedirs(self.summary_folder, exist_ok=True)
            sum_writer = SummaryWriter(self.summary_folder)

        cnt = -1 # total batch count
        avg_batch_loss_mllp = 0.0
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        best_metric = float('-inf')
        for epo in range(1, self.epoch + 1):
            if self.lr_decay_epoch is not None:
                optimizer = self.exp_lr_scheduler(
                    optimizer, epo, init_lr=self.lr,
                    lr_decay_rate=self.lr_decay_rate, lr_decay_epoch=self.lr_decay_epoch)

            epoch_loss_mllp = 0.0
            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0
            metric_histc = []
            ba_cnt = 0

            for X, y in train_loader:
                ba_cnt += 1
                cnt += 1
                if X.is_sparse:
                    X = X.to_dense()
                X = X.to(self.device, non_blocking=self.pin_memory)
                y = nn.functional.one_hot(y, num_classes=self.n_classes).to(self.device, non_blocking=self.pin_memory)

                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred_mllp, y_pred_rrl = self.net.forward(X)
                with torch.no_grad():
                    y_prob = torch.softmax(y_pred_rrl, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss_mllp = criterion(y_pred_mllp, y_arg)
                    loss_rrl = criterion(y_pred_rrl, y_arg)
                    ba_loss_mllp = loss_mllp.item()
                    ba_loss_rrl = loss_rrl.item()
                    epoch_loss_mllp += ba_loss_mllp
                    epoch_loss_rrl += ba_loss_rrl
                    avg_batch_loss_mllp += ba_loss_mllp
                    avg_batch_loss_rrl += ba_loss_rrl
                y_pred_mllp.backward((y_prob - y) / y.shape[0])  # for CrossEntropy Loss
                optimizer.step()
                for i, param in enumerate(self.net.parameters()):
                    abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                    abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if cnt % self.log_step == 0 and cnt != 0 and sum_writer is not None:
                    avg_batch_loss_mllp /= self.log_step
                    avg_batch_loss_rrl /= self.log_step
                    sum_writer.add_scalar('Avg_Batch_Loss_MLLP', avg_batch_loss_mllp, cnt)
                    sum_writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl, cnt)
                    self.logger_info(f'epoch {epo} / {self.epoch}; batch {ba_cnt} / {len(train_loader)}; loss_mllp = {avg_batch_loss_mllp}; loss_rrl = {avg_batch_loss_rrl}')
                    avg_batch_loss_mllp = 0.0
                    avg_batch_loss_rrl = 0.0

                if cnt % self.eval_step == 0:
                    self.net.eval()
                    if val_loader is not None:
                        metric_dict, metric_dict_b = self.cal_metric_with_dataloader(val_loader)
                    else:
                        metric_dict, metric_dict_b = self.cal_metric_with_dataloader(train_loader)
                    cur_metric = flatten_dict(metric_dict_b, sep=' ')[self.eval_metric]
                    metric_histc.append(cur_metric)
                    self.logger_info(f'epoch {epo} / {self.epoch}; batch {ba_cnt} / {len(train_loader)}; {self.eval_metric} = {cur_metric}')
                    if self.save_mode == 'best' and cur_metric > best_metric:
                        self.logger_info(f'update best {self.eval_metric}: {best_metric} -> {cur_metric}')
                        best_metric = cur_metric
                        self.detect_dead_node(train_loader) # TODO: accelerrate
                        self.save()
                    if sum_writer is not None:
                        sum_writer.add_scalar('Accuracy_MLLP', metric_dict['accuracy'], cnt)
                        sum_writer.add_scalar('Accuracy_RRL', metric_dict_b['accuracy'], cnt)
                        sum_writer.add_scalar('F1_Score_MLLP', metric_dict['macro avg']['f1-score'], cnt)
                        sum_writer.add_scalar('F1_Score_RRL', metric_dict_b['macro avg']['f1-score'], cnt)
                    self.net.train()

            epoch_loss_mllp /= ba_cnt
            epoch_loss_rrl /= ba_cnt
            abs_gradient_avg /= ba_cnt
            self.logger_info(f'epoch: {epo} / {self.epoch}, loss_mllp: {epoch_loss_mllp}, loss_rrl: {epoch_loss_rrl}; '
                             f'{self.eval_metric}: {np.mean(metric_histc) if metric_histc else np.nan}')
            for name, param in self.net.named_parameters():
                maxl = 1 if 'con_layer' in name or 'dis_layer' in name else 0
                epoch_histc[name].append(torch.histc(param.data, bins=10, max=maxl).cpu().numpy())
            if sum_writer is not None:
                sum_writer.add_scalar('Training_Loss_MLLP', epoch_loss_mllp, epo)
                sum_writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                sum_writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                sum_writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)

        if self.save_mode == 'end':
            self.detect_dead_node(train_loader)
            self.save()

        self.net.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.remove_logger()
        return epoch_histc


    def cal_metric_with_dataloader(self, dataloader):
        """
        Args:
            dataloader (DataLoader)
        Returns:
            dict: metric dict (continueous net). The format is the same as return of function get_cls_metric_dict.
            dict: metric dict (binary net). The format is the same as return of function get_cls_metric_dict.
        """
        y_true, y_pred, y_pred_b = [], [], []
        with torch.no_grad():
            if self.verbose:
                dataloader = tqdm(dataloader)
            for X, y in dataloader:
                if X.is_sparse:
                    X = X.to_dense()
                y_true.append(y.cpu().numpy())
                X = X.to(self.device)
                logits, logits_b = self.net.forward(X)
                y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
                y_pred_b.append(torch.argmax(logits_b, dim=1).cpu().numpy())

        y_true = self.classes_[np.hstack(y_true)]
        y_pred = self.classes_[np.hstack(y_pred)]
        y_pred_b = self.classes_[np.hstack(y_pred_b)]
        metric_dict = get_cls_metric_dict(y_true, y_pred, label_list=self.classes_, cal_confusion=True)
        metric_dict_b = get_cls_metric_dict(y_true, y_pred_b, label_list=self.classes_, cal_confusion=True)
        return metric_dict, metric_dict_b


    def predict(self, X):
        """
        Args:
            X (array-like): (n_samples, n_features)
        Returns:
            np.ndarray: (n_samples,)
        """
        log_proba = self.predict_log_proba(X)
        y_pred = np.argmax(log_proba, axis=1)
        if self.need_to_map_y:
            return self.classes_[y_pred]
        return y_pred


    def predict_log_proba(self, X):
        """
        Args:
            X (array-like): (n_samples, n_features)
        Returns:
            np.ndarray: (n_samples, n_classes)
        """
        self.net.eval()
        self.net.to(self.device)
        if self.feat_order_manager is not None:
            self.feat_order_manager.transform(X, inplace=True)

        n_samples = X.shape[0]
        if n_samples < self.batch_size:
            if sp.issparse(X):
                X = np.array(X)
            _, logits_b = self.net.forward(torch.tensor(X).to(self.device))
            logits_b = logits_b.cpu().numpy()
        else:
            dataloader = self.convert_mat_to_dataloader(X, batch_size=self.batch_size, shuffle=False)
            logits_b = []
            for X_batch in dataloader:
                X_batch = X_batch[0]
                if X_batch.is_sparse:
                    X_batch = X_batch.to_dense()
                _, logits_b_batch = self.net.forward(X_batch.to(self.device))
                logits_b.append(logits_b_batch.cpu().numpy())
            logits_b = np.vstack(logits_b)

        if self.feat_order_manager is not None:
            self.feat_order_manager.inverse_transform(X, inplace=True)
        return logits_b


    def predict_proba(self, X):
        """
        Args:
            X (array-like): (n_samples, n_features)
        Returns:
            np.ndarray: (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)


    def get_save_ignore_attrs(self):
        return ['net', 'save_folder', 'model_path', 'log_path', 'summary_folder']


    def save(self, save_folder=None):
        if save_folder:
            self.init_save_path(save_folder)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        ignore_attrs = self.get_save_ignore_attrs()
        attr_dict = {k: v for k, v in self.__dict__.items() if k not in ignore_attrs}
        torch.save({'attr_dict': attr_dict, 'state_dict': self.net.state_dict()}, self.model_path)
        self.logger_info(f'{self.__class__.__name__} saved in {self.model_path}')


    def load(self, save_folder=None):
        if save_folder:
            self.init_save_path(save_folder)
        obj_dict = torch.load(self.model_path, map_location='cpu')
        attr_dict = obj_dict['attr_dict']
        state_dict = obj_dict['state_dict']
        for k, v in attr_dict.items():
            setattr(self, k, v)
        self.net = self.init_network()
        self.net.load_state_dict(state_dict)


    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device)
                layer.forward_tot = 0
            for x, y in data_loader:
                if x.is_sparse:
                    x = x.to_dense()
                x = x.to(self.device)
                x_res = None
                for i, layer in enumerate(self.net.layer_list[:-1]):
                    if i <= 1:
                        x = layer.binarized_forward(x)
                    else:
                        x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                        x_res = x
                        x = layer.binarized_forward(x_cat)
                    layer.node_activation_cnt += torch.sum(x, dim=0)
                    layer.forward_tot += x.shape[0]


    def extract_rules(self, feature_names, data_loader=None, weight_sort=None, cal_act_rate=False,
                      need_bias=False, and_op=AND_OP, or_op=OR_OP, not_op=NOT_OP):
        """
        Args:
            feature_names (array-like): (n_features,)
            data_loader (DataLoader): Used to detect dead node
            value_convert_func (dict): {feature_name: function}; function: float -> object
            weight_sort (str or int): feature name. The rules will be sorted by the final LR layer weight which refers to the feature.
            cal_act_rate (bool): whether the return includes 'ACT_NODE', 'TOTAL_NODE', 'ACT_RATE'
            need_bias (bool): whether the return includes '{label}_BIAS'
        Returns:
            list: [{
                    'RULE_ID': (int, int), # (prev_layer_id, output_id)
                    '{label}_WIEGHT': float, # rule weight for specific label
                    '{label}_BIAS': float, # bias for specific label; optional
                    'RULE_DESC': str, # description of rule
                    'RULE_PARSE': tuple, # (op, [para1, (op, [para21, para22, ...]), para3, ...])
                    'ACT_NODE': int, # optional
                    'TOTAL_NODE': int, # optional
                    'ACT_RATE': float, # optional
            }, ...]
        """
        if self.net.layer_list[1].node_activation_cnt is None:
            assert data_loader is not None, 'Need train_loader for the dead nodes detection'
            self.detect_dead_node(data_loader)

        if isinstance(feature_names, list):
            feature_names = np.array(feature_names)
        if self.feat_order_manager is not None:
            feature_names = self.feat_order_manager.transform(feature_names)

        rule_leaf_names = self.net.layer_list[0].get_rule_leaf_names(feature_names, not_op=not_op)
        self.net.layer_list[1].get_rules(self.net.layer_list[0], None)
        self.net.layer_list[1].get_rule_description((None, rule_leaf_names), and_op=and_op, or_op=or_op)
        self.net.layer_list[1].get_rule_parse_objs((None, rule_leaf_names), and_op=and_op, or_op=or_op)

        if len(self.net.layer_list) >= 4:
            self.net.layer_list[2].get_rules(self.net.layer_list[1], None)
            self.net.layer_list[2].get_rule_description((None, self.net.layer_list[1].rule_name), wrap=True, and_op=and_op, or_op=or_op)
            self.net.layer_list[2].get_rule_parse_objs((None, self.net.layer_list[1].rule_parse_objs), wrap=True, and_op=and_op, or_op=or_op)

        if len(self.net.layer_list) >= 5:
            for i in range(3, len(self.net.layer_list) - 1):
                self.net.layer_list[i].get_rules(self.net.layer_list[i-1], self.net.layer_list[i-2])
                self.net.layer_list[i].get_rule_description(
                    (self.net.layer_list[i-2].rule_name, self.net.layer_list[i-1].rule_name), wrap=True, and_op=and_op, or_op=or_op)
                self.net.layer_list[i].get_rule_parse_objs(
                    (self.net.layer_list[i-2].rule_parse_objs, self.net.layer_list[i-1].rule_parse_objs), wrap=True, and_op=and_op, or_op=or_op)

        prev_layer = self.net.layer_list[-2]
        skip_connect_layer = self.net.layer_list[-3]
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        if skip_connect_layer.layer_type == 'union':
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        else:
            merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

        Wl, bl = list(self.net.layer_list[-1].parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float)) # {(prev_layer_offset, output_id): {label_id: weight, ...}, ...}
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        ret_list = []
        for rid, v in marked.items():
            now_layer = self.net.layer_list[-1 + rid[0]]
            rule_dict = {
                'RULE_ID': rid,
                'RULE_DESC': now_layer.rule_name[rid[1]],
                'RULE_PARSE': now_layer.rule_parse_objs[rid[1]],
            }
            for i, label_name in enumerate(self.classes_):
                rule_dict[f'{label_name}_WEIGHT'] = float(v[i])
                if need_bias:
                    rule_dict[f'{label_name}_BIAS'] = float(bl[i])
            if cal_act_rate:
                rule_dict['ACT_NODE'] = float(now_layer.node_activation_cnt[rid2dim[rid]])
                rule_dict['TOTAL_NODE'] = float(now_layer.forward_tot)
                rule_dict['ACT_RATE'] = rule_dict['ACT_NODE'] / rule_dict['TOTAL_NODE']
            ret_list.append(rule_dict)

        if weight_sort is not None:
            ret_list.sort(key=lambda d: d[f'{weight_sort}_WEIGHT'], reverse=True)
        return ret_list


if __name__ == '__main__':
    pass


