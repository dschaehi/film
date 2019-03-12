#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vr.models.layers import init_modules, build_stem, build_relational_module, build_multimodal_core, build_classifier, ResidualBlock
from vr.models.filmed_net import coord_map
from vr.embedding import expand_embedding_vocab


class StackedAttention(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(StackedAttention, self).__init__()
    self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
    self.Wu = nn.Linear(input_dim, hidden_dim)
    self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
    self.hidden_dim = hidden_dim
    self.attention_maps = None
    init_modules(self.modules(), init='normal')

  def forward(self, v, u):
    """
    Input:
    - v: N x D x H x W
    - u: N x D

    Returns:
    - next_u: N x D
    """
    N, K = v.size(0), self.hidden_dim
    D, H, W = v.size(1), v.size(2), v.size(3)
    v_proj = self.Wv(v) # N x K x H x W
    u_proj = self.Wu(u) # N x K
    u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
    h = F.tanh(v_proj + u_proj_expand)
    p = F.softmax(self.Wp(h).view(N, H * W), dim=1).view(N, 1, H, W)
    self.attention_maps = p.data.clone()

    v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
    next_u = u + v_tilde
    return next_u


class LstmEncoder(nn.Module):
  def __init__(self, token_to_idx, wordvec_dim=300,
               rnn_dim=256, rnn_num_layers=2, rnn_dropout=0):
    super(LstmEncoder, self).__init__()
    self.token_to_idx = token_to_idx
    self.NULL = token_to_idx['<NULL>']
    self.START = token_to_idx['<START>']
    self.END = token_to_idx['<END>']

    self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
    self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                       dropout=rnn_dropout, batch_first=True)

  def expand_vocab(self, token_to_idx, word2vec=None, std=0.01):
    expand_embedding_vocab(self.embed, token_to_idx,
                           word2vec=word2vec, std=std)

  def forward(self, x):
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence
    x_cpu = x.data.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data).long()
    idx = Variable(idx, requires_grad=False)

    hs, _ = self.rnn(self.embed(x))
    idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
    H = hs.size(2)
    return hs.gather(1, idx).view(N, H)


def build_cnn(feat_dim=(1024, 14, 14),
              res_block_dim=128,
              num_res_blocks=0,
              proj_dim=512,
              pooling='maxpool2'):
  C, H, W = feat_dim
  layers = []
  if num_res_blocks > 0:
    layers.append(nn.Conv2d(C, res_block_dim, kernel_size=3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    C = res_block_dim
    for _ in range(num_res_blocks):
      layers.append(ResidualBlock(C))
  if proj_dim > 0:
    layers.append(nn.Conv2d(C, proj_dim, kernel_size=1, padding=0))
    layers.append(nn.ReLU(inplace=True))
    C = proj_dim
  if pooling == 'maxpool2':
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    H, W = H // 2, W // 2
  elif pooling == 'maxpoolfull':
    if H != W:
      assert(NotImplementedError)
    layers.append(nn.MaxPool2d(kernel_size=H, stride=H, padding=0))
    H = W = 1
  return nn.Sequential(*layers), (C, H, W)


def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
  layers = []
  D = input_dim
  if dropout > 0:
    layers.append(nn.Dropout(p=dropout))
  if use_batchnorm:
    layers.append(nn.BatchNorm1d(input_dim))
  for dim in hidden_dims:
    layers.append(nn.Linear(D, dim))
    if use_batchnorm:
      layers.append(nn.BatchNorm1d(dim))
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
    layers.append(nn.ReLU(inplace=True))
    D = dim
  layers.append(nn.Linear(D, output_dim))
  return nn.Sequential(*layers)


class CnnModel(nn.Module):
  def __init__(self, vocab,
               feature_dim=(1024, 14, 14), stem_module_dim=128,
               stem_use_resnet=False, stem_resnet_fixed=False, resnet_model_stage=3,
               stem_num_layers=2, stem_batchnorm=False, stem_kernel_size=3,
               stem_stride=1, stem_stride2_freq=0, stem_padding=None,
               cnn_res_block_dim=128, cnn_num_res_blocks=0, cnn_proj_dim=512, cnn_pooling='maxpool2',
               fc_dims=(1024,), fc_use_batchnorm=False, fc_dropout=0):
    super(CnnModel, self).__init__()
    self.stem = build_stem(stem_use_resnet, stem_resnet_fixed, feature_dim[0], stem_module_dim,
                           resnet_model_stage=resnet_model_stage, num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, stride2_freq=stem_stride2_freq, padding=stem_padding)

    if stem_stride2_freq > 0:
      module_H = feature_dim[1] // (2 ** (stem_num_layers // stem_stride2_freq))
      module_W = feature_dim[2] // (2 ** (stem_num_layers // stem_stride2_freq))
    else:
      module_H = feature_dim[1]
      module_W = feature_dim[2]

    self.cnn, (C, H, W) = build_cnn(
      feat_dim=(stem_module_dim, module_H, module_W), res_block_dim=cnn_res_block_dim,
      num_res_blocks=cnn_num_res_blocks, proj_dim=cnn_proj_dim, pooling=cnn_pooling
    )
    module_C = C * H * W

    self.classifier = build_classifier(
      module_C=module_C, module_H=None, module_W=None, num_answers=len(vocab['answer_token_to_idx']),
      fc_dims=fc_dims, proj_dim=None, downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )

  def forward(self, questions, feats):
    feats = self.stem(feats)
    N, C, H, W = feats.size()
    feats = self.cnn(feats)
    feats = feats.view(N, -1)
    scores = self.classifier(feats)
    return scores


class LstmModel(nn.Module):
  def __init__(self, vocab,
               rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
               fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):
    super(LstmModel, self).__init__()
    rnn_kwargs = {
      'token_to_idx': vocab['question_token_to_idx'],
      'wordvec_dim': rnn_wordvec_dim,
      'rnn_dim': rnn_dim,
      'rnn_num_layers': rnn_num_layers,
      'rnn_dropout': rnn_dropout,
    }
    self.rnn = LstmEncoder(**rnn_kwargs)

    self.classifier = build_classifier(
      module_C=rnn_dim, module_H=None, module_W=None,
      num_answers=len(vocab['answer_token_to_idx']), fc_dims=fc_dims, proj_dim=None,
      downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )

  def forward(self, questions, feats):
    q_feats = self.rnn(questions)
    scores = self.classifier(q_feats)
    return scores


class CnnLstmModel(nn.Module):
  def __init__(self, vocab,
               rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
               feature_dim=(1024, 14, 14), stem_module_dim=128,
               stem_use_resnet=False, stem_resnet_fixed=False, resnet_model_stage=3,
               stem_num_layers=2, stem_batchnorm=False, stem_kernel_size=3,
               stem_stride=1, stem_stride2_freq=0, stem_padding=None,
               relational_module=False, rel_module_dim=256, rel_num_layers=4,
               multimodal_core=False, mc_module_dim=256, mc_num_layers=4, mc_batchnorm=True,
               cnn_res_block_dim=128, cnn_num_res_blocks=0, cnn_proj_dim=512, cnn_pooling='maxpool2',
               fc_dims=(1024,), fc_use_batchnorm=False, fc_dropout=0):
    super(CnnLstmModel, self).__init__()
    rnn_kwargs = {
      'token_to_idx': vocab['question_token_to_idx'],
      'wordvec_dim': rnn_wordvec_dim,
      'rnn_dim': rnn_dim,
      'rnn_num_layers': rnn_num_layers,
      'rnn_dropout': rnn_dropout,
    }
    self.rnn = LstmEncoder(**rnn_kwargs)

    self.stem = build_stem(stem_use_resnet, stem_resnet_fixed, feature_dim[0], stem_module_dim,
                           resnet_model_stage=resnet_model_stage, num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, stride2_freq=stem_stride2_freq, padding=stem_padding)

    if stem_stride2_freq > 0:
      module_H = feature_dim[1] // (2 ** (stem_num_layers // stem_stride2_freq))
      module_W = feature_dim[2] // (2 ** (stem_num_layers // stem_stride2_freq))
    else:
      module_H = feature_dim[1]
      module_W = feature_dim[2]

    assert not relational_module or not multimodal_core
    self.relational_module = relational_module
    self.multimodal_core = multimodal_core

    if self.relational_module:
      # https://arxiv.org/abs/1706.01427
      self.coords = coord_map((module_H, module_W))
      self.rel = build_relational_module(
        feature_dim=((stem_module_dim + 2) * 2 + rnn_dim), module_dim=rel_module_dim,
        num_layers=rel_num_layers
      )
      module_C = rel_module_dim

    elif self.multimodal_core:
      # https://arxiv.org/abs/1809.04482
      self.mc = build_multimodal_core(
        feature_dim=(stem_module_dim + rnn_dim), module_dim=mc_module_dim, num_layers=mc_num_layers,
        with_batchnorm=mc_batchnorm
      )
      module_C = mc_module_dim

    else:
      self.cnn, (C, H, W) = build_cnn(
        feat_dim=(stem_module_dim, module_H, module_W), res_block_dim=cnn_res_block_dim,
        num_res_blocks=cnn_num_res_blocks, proj_dim=cnn_proj_dim, pooling=cnn_pooling
      )
      module_C = C * H * W + rnn_dim

    self.classifier = build_classifier(
      module_C=module_C, module_H=None, module_W=None, num_answers=len(vocab['answer_token_to_idx']),
      fc_dims=fc_dims, proj_dim=None, downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )

  def forward(self, questions, feats):
    feats = self.stem(feats)
    q_feats = self.rnn(questions)
    N, C, H, W = feats.size()
    N1, Q = q_feats.size()
    assert N == N1

    if self.relational_module:
      # Code adapted from https://github.com/kimhc6028/relational-networks
      feats = torch.cat([feats, self.coords.unsqueeze(0).repeat(N, 1, 1, 1)], 1)
      feats = feats.view(N, C + 2, H * W).permute(0, 2, 1)
      feats1 = feats.unsqueeze(1)
      feats1 = feats1.repeat(1, H * W, 1, 1)
      feats2 = torch.cat([feats, q_feats.unsqueeze(1).repeat(1, H * W, 1)], 2)
      feats2 = feats2.unsqueeze(2)
      feats2 = feats2.repeat(1, 1, H * W, 1)
      feats = torch.cat([feats1, feats2], 3)
      feats = feats.view(N * H * W * H * W, (C + 2) * 2 + Q)
      feats = self.rel(feats)
      feats = feats.view(N, H * W * H * W, feats.size(1))
      feats = feats.sum(1)

    elif self.multimodal_core:
      q_feats = q_feats.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
      feats = torch.cat([feats, q_feats], 1)
      feats = self.mc(feats)
      feats = feats.sum(3).sum(2)

    else:
      feats = self.cnn(feats)
      feats = torch.cat([q_feats, feats.view(N, -1)], 1)

    scores = self.classifier(feats)
    return scores


class CnnLstmSaModel(nn.Module):
  def __init__(self, vocab,
               rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
               feature_dim=(1024, 14, 14), stem_module_dim=128,
               stem_use_resnet=False, stem_resnet_fixed=False, resnet_model_stage=3,
               stem_num_layers=2, stem_batchnorm=False, stem_kernel_size=3,
               stem_stride=1, stem_stride2_freq=0, stem_padding=None,
               stacked_attn_dim=512, num_stacked_attn=2,
               fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):
    super(CnnLstmSaModel, self).__init__()
    rnn_kwargs = {
      'token_to_idx': vocab['question_token_to_idx'],
      'wordvec_dim': rnn_wordvec_dim,
      'rnn_dim': rnn_dim,
      'rnn_num_layers': rnn_num_layers,
      'rnn_dropout': rnn_dropout,
    }
    self.rnn = LstmEncoder(**rnn_kwargs)

    self.stem = build_stem(stem_use_resnet, stem_resnet_fixed, feature_dim[0], stem_module_dim,
                           resnet_model_stage=resnet_model_stage, num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, stride2_freq=stem_stride2_freq, padding=stem_padding)

    if stem_stride2_freq > 0:
      module_H = feature_dim[1] // (2 ** (stem_num_layers // stem_stride2_freq))
      module_W = feature_dim[2] // (2 ** (stem_num_layers // stem_stride2_freq))
    else:
      module_H = feature_dim[1]
      module_W = feature_dim[2]

    self.image_proj = nn.Conv2d(stem_module_dim, rnn_dim, kernel_size=1, padding=0)
    self.stacked_attns = []
    for i in range(num_stacked_attn):
      sa = StackedAttention(rnn_dim, stacked_attn_dim)
      self.stacked_attns.append(sa)
      self.add_module('stacked-attn-%d' % i, sa)

    self.classifier = build_classifier(
      module_C=rnn_dim, module_H=None, module_W=None,
      num_answers=len(vocab['answer_token_to_idx']), fc_dims=fc_dims, proj_dim=None,
      downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )
    init_modules(self.modules(), init='normal')

  def forward(self, questions, feats):
    u = self.rnn(questions)
    feats = self.stem(feats)
    v = self.image_proj(feats)

    for sa in self.stacked_attns:
      u = sa(v, u)

    scores = self.classifier(u)
    return scores
