#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
import torchvision

class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
    self.with_batchnorm = with_batchnorm
    if with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim)
      self.bn2 = nn.BatchNorm2d(out_dim)
    self.with_residual = with_residual
    if in_dim == out_dim or not with_residual:
      self.proj = None
    else:
      self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

  def forward(self, x):
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out


class ConcatBlock(nn.Module):
  def __init__(self, num_inputs, dim, with_residual=True, with_batchnorm=True):
    super(ConcatBlock, self).__init__()
    self.proj = nn.Conv2d(num_inputs * dim, dim, kernel_size=1, padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                        with_batchnorm=with_batchnorm)

  def forward(self, *xs):
    out = torch.cat(xs, 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.res_block(out)
    return out


class GlobalAveragePool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(2).squeeze(2)


class GlobalSumPool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).sum(2).squeeze(2)


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


def build_stem(use_resnet, resnet_fixed, feature_dim, module_dim, resnet_model_stage=3, num_layers=2,
               with_batchnorm=True, kernel_size=3, stride=1, stride2_freq=0, padding=None):
  if use_resnet:
    if not hasattr(torchvision.models, 'resnet101'):
      raise ValueError('Invalid model "resnet101"')
    loaded_model = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [
      loaded_model.conv1,
      loaded_model.bn1,
      loaded_model.relu,
      loaded_model.maxpool
    ]
    for i in range(resnet_model_stage):
      name = 'layer%d' % (i + 1)
      layers.append(getattr(loaded_model, name))
    # Freeze parameters of ResNet (fixed ResNet)
    if resnet_fixed:
      for layer in layers:
        for param in layer.parameters():
          param.requires_grad = False
    prev_dim = feature_dim
    if padding is None:
      if kernel_size % 2 == 0:
        raise(NotImplementedError)
      padding = kernel_size // 2
    for i in range(num_layers):
      layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding))
      if with_batchnorm:
        layers.append(nn.BatchNorm2d(module_dim))
      layers.append(nn.ReLU(inplace=True))
      prev_dim = module_dim
  else: # custom CNN
    layers = []
    prev_dim = feature_dim
    if padding is None:  # Calculate default padding when None provided
      if kernel_size % 2 == 0:
        raise NotImplementedError
      padding = kernel_size // 2
    for i in range(num_layers):
      if stride2_freq > 0 and (i + 1) % stride2_freq == 0:
        stride = 2
      else:
        stride = 1
      layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding))
      if with_batchnorm:
        layers.append(nn.BatchNorm2d(module_dim))
      layers.append(nn.ReLU(inplace=True))
      prev_dim = module_dim
  return nn.Sequential(*layers)


def build_relational_module(feature_dim, module_dim=256, num_layers=4):
  layers = []
  prev_dim = feature_dim
  for i in range(num_layers):
    layers.append(nn.Linear(prev_dim, module_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = module_dim
  return nn.Sequential(*layers)


def build_multimodal_core(feature_dim, module_dim=256, num_layers=4, with_batchnorm=True, kernel_size=1):
  layers = []
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(feature_dim))
  prev_dim = feature_dim
  for i in range(num_layers):
    layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = module_dim
  return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=()):
  layers = []
  if module_H is None and module_W is None:
    module_H = module_W = 1
  if proj_dim is not None and proj_dim > 0:
    layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(proj_dim))
    layers.append(nn.ReLU(inplace=True))
    module_C = proj_dim
  if downsample is not None and ('maxpool' in downsample or 'avgpool' in downsample):
    pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
    if 'full' in downsample:
      # if module_H != module_W:
      #   raise NotImplementedError
      pool_size = (module_H, module_W)
    else:
      pool_size = (int(downsample[-1]), int(downsample[-1]))
    # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
    padding = 0 if ((module_H % pool_size[0] == 0) and (module_W % pool_size[1] == 0)) else 1
    layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
    module_H = math.ceil(module_H / pool_size[0])
    module_W = math.ceil(module_W / pool_size[1])
  if downsample == 'aggressive':
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.AvgPool2d(kernel_size=module_H // 2, stride=module_W // 2))
    module_H = module_W = 1
    assert fc_dims == []  # No FC layers here
  if downsample is not None:
    layers.append(Flatten())
  prev_dim = module_C * module_H * module_W
  if isinstance(dropout, (int, float)):
    dropout = [dropout for _ in fc_dims]
  for next_dim, p in zip(fc_dims, dropout):
    layers.append(nn.Linear(prev_dim, next_dim))
    if with_batchnorm:
      layers.append(nn.BatchNorm1d(next_dim))
    layers.append(nn.ReLU(inplace=True))
    if p > 0.0:
      layers.append(nn.Dropout(p=p))
    prev_dim = next_dim
  layers.append(nn.Linear(prev_dim, num_answers))
  return nn.Sequential(*layers)


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = kaiming_normal
  elif init.lower() == 'uniform':
    init_params = kaiming_uniform
  else:
    return
  for m in modules:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      init_params(m.weight)
