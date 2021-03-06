# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reference: https://github.com/12wang3/rrl

import torch
import torch.nn as nn
from collections import defaultdict

from rrl.constant import EPSILON, THRESHOLD, INIT_RANGE
from rrl.constant import AND_OP, OR_OP, NOT_OP, GT_OP, LT_OP


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, use_not=False, left=None, right=None, dtype=None):
        super(BinarizeLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}

        if isinstance(left, float):
            left = torch.tensor([left])
        if isinstance(right, float):
            right = torch.tensor([right])
        self.register_buffer('left', left)
        self.register_buffer('right', right)

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1], dtype=dtype) * (self.right - self.left)
                cr = self.left + torch.rand(self.n, self.input_dim[1], dtype=dtype) * (self.right - self.left)
            else:
                cl = 3. * (2. * torch.rand(self.n, self.input_dim[1], dtype=dtype) - 1.)
                cr = 3. * (2. * torch.rand(self.n, self.input_dim[1], dtype=dtype) - 1.)
            self.register_buffer('cl', cl)
            self.register_buffer('cr', cr)

    def forward(self, x):
        if self.input_dim[1] > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            x = x.unsqueeze(-1)
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
            return torch.cat((x_disc, Binarize.apply(x - self.cl.t()).view(x.shape[0], -1),
                              1 - Binarize.apply(x - self.cr.t()).view(x.shape[0], -1)), dim=1)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            return self.forward(x)

    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(self.cl.data > self.right, self.right, self.cl.data)
            self.cl.data = torch.where(self.cl.data < self.left, self.left, self.cl.data)

            self.cr.data = torch.where(self.cr.data > self.right, self.right, self.cr.data)
            self.cr.data = torch.where(self.cr.data < self.left, self.left, self.cr.data)


    def get_rule_leaf_names(self, feature_name, gt_op=GT_OP, lt_op=LT_OP, not_op=NOT_OP):
        """
        Args:
            feature_name (array-like): length = input dim of binarizer layer
        Returns:
            list: [str, str, ...]; length = output dim of binarizer layer
        """
        ret_names = []
        for i in range(self.input_dim[0]):
            ret_names.append(feature_name[i])
        if self.use_not:
            for i in range(self.input_dim[0]):
                ret_names.append(not_op + feature_name[i])
        if self.input_dim[1] > 0:
            for c, op in [(self.cl, gt_op), (self.cr, lt_op)]:
                c = c.detach().cpu().numpy()
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i]
                    for j in ci:
                        ret_names.append('{} {} {:.4f}'.format(fi_name, op, j))
        return ret_names


class Product(torch.autograd.Function):
    """Tensor product function."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim, dtype=None):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'

        self.fc1 = nn.Linear(self.input_dim, self.output_dim, dtype=dtype)

    def forward(self, x):
        return self.fc1(x)

    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False, dtype=None):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim, dtype=dtype))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return self.Product.apply(1 - (1 - x).unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False, dtype=None):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim, dtype=dtype))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - self.Product.apply(1 - x.unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []
    Wb = (layer.W > 0.5).type(torch.int).detach().cpu().numpy()

    if skip_connect_layer is not None:
        shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
        prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
    else:
        merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

    for ri, row in enumerate(Wb):
        if layer.node_activation_cnt[ri + pos_shift] == 0 or layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot:
            dim2id[ri + pos_shift] = -1
            continue
        rule = {}
        bound = {}
        if prev_layer.layer_type == 'binarization' and prev_layer.input_dim[1] > 0:
            c = torch.cat((prev_layer.cl.t().reshape(-1), prev_layer.cr.t().reshape(-1))).detach().cpu().numpy()
        for i, w in enumerate(row):
            if w > 0 and merged_dim2id[i][1] != -1:
                if prev_layer.layer_type == 'binarization' and i >= prev_layer.disc_num:
                    ci = i - prev_layer.disc_num
                    bi = ci // prev_layer.n
                    if bi not in bound:
                        bound[bi] = [i, c[ci]]
                        rule[(-1, i)] = 1
                    else:
                        if (ci < c.shape[0] // 2 and layer.layer_type == 'conjunction') or \
                           (ci >= c.shape[0] // 2 and layer.layer_type == 'disjunction'):
                            func = max
                        else:
                            func = min
                        bound[bi][1] = func(bound[bi][1], c[ci])
                        if bound[bi][1] == c[ci]:
                            del rule[(-1, bound[bi][0])]
                            rule[(-1, i)] = 1
                            bound[bi][0] = i
                else:
                    rule[merged_dim2id[i]] = 1
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list


class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False, dtype=None):
        super(UnionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None # ([con_rule_path1, ...], [dis_rule_path1, ...]); rule_path = [((prev_offset1, output_id1), ...), ...]
        self.rule_name = None # [str, str, ...]; length=output_dim
        self.rule_parse_objs = None # [tuple, tuple, ...]; lengh=output_dim

        self.con_layer = ConjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad, dtype=dtype)
        self.dis_layer = DisjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad, dtype=dtype)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat([self.con_layer.binarized_forward(x),
                          self.dis_layer.binarized_forward(x)], dim=1)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = self.node_activation_cnt

        con_dim2id, con_rule_list = extract_rules(prev_layer, skip_connect_layer, self.con_layer)
        dis_dim2id, dis_rule_list = extract_rules(prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[0])

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})
        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

        return dim2id, rule_list


    def get_rule_description(self, prev, wrap=False, and_op=AND_OP, or_op=OR_OP):
        """
        Args:
            prev (tuple): (..., prev_prev_rule_names, prev_rule_names); rule_names = [rule_str, ...], rule_str = e.g. '7_x & 8_x & 9_x'
            wrap (bool): whether to use bracket wrapper
        Returns:
            list: [rule_str1, rule_str2, ...]
        """
        self.rule_name = []
        prev_len = len(prev)
        for rl, op in zip(self.rule_list, (and_op, or_op)):
            for rule in rl:
                name = ''
                for i, ri in enumerate(rule):
                    op_str = ' {} '.format(op) if i != 0 else ''
                    var_str = ('({})' if wrap else '{}').format(prev[prev_len + ri[0]][ri[1]])
                    name += op_str + var_str
                self.rule_name.append(name)
        return self.rule_name


    def get_rule_parse_objs(self, prev, and_op=AND_OP, or_op=OR_OP):
        """format of rule parse objs: (op, [para1, (op, [para21, para22, ...]), para3, ...])
        Args:
            prev (tuple): (..., prev_prev_rule_parse_objs, prev_rule_parse_objs); rule_parse_objs = [rule_parse_obj1, rule_parse_obj2, ...]; rule_parse_obj = e.g. ('&', ['7_x', '8_x', '9_x'])
        Returns:
            list: [rule_parse_obj1, rule_parse_obj2, ...]
        """
        self.rule_parse_objs = []
        prev_len = len(prev)
        for rule_paths, op in zip(self.rule_list, (and_op, or_op)):
            for rule_path in rule_paths:
                paras = [prev[prev_len + prev_offset][output_id] for prev_offset, output_id in rule_path]
                self.rule_parse_objs.append((op, paras))
        return self.rule_parse_objs


if __name__ == '__main__':
    pass

