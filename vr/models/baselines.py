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
from vr.models.filmed_net import coord_map, FiLM
from vr.embedding import expand_embedding_vocab


class StackedAttention(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size=1, film=False):
    super(StackedAttention, self).__init__()
    self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
    # self.Wv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0)
    # self.Wu = nn.Linear(input_dim, hidden_dim)
    if film:
      self.Wu = nn.Linear(input_dim, 2 * hidden_dim)
      self.film = FiLM()
    else:
      self.Wu = nn.Linear(input_dim, hidden_dim)
      self.film = None
    self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2)
    self.hidden_dim = hidden_dim
    self.attention_maps = None
    init_modules(self.modules(), init='normal')

  def forward(self, v, u):
  # def forward(self, v, u, u1):
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
    if self.film is None:
      u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
      h = F.tanh(v_proj + u_proj_expand)
    else:
      h = F.tanh(self.film(x=v_proj, gammas=u_proj[:, :self.hidden_dim], betas=u_proj[:, self.hidden_dim:]))
    p = F.softmax(self.Wp(h).view(N, H * W), dim=1).view(N, 1, H, W)
    self.attention_maps = p.data.clone()

    v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
    next_u = u + v_tilde
    # next_u = u1 + v_tilde
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


class CnnModel(nn.Module):
  def __init__(self, vocab,
               feature_dim=(1024, 14, 14), stem_module_dim=128,
               stem_use_resnet=False, stem_resnet_fixed=False, resnet_model_stage=3,
               stem_num_layers=2, stem_batchnorm=False, stem_kernel_size=3,
               stem_stride=1, stem_stride2_freq=0, stem_padding=None,
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

    self.conv = nn.Conv2d(stem_module_dim, stem_module_dim, kernel_size=1, padding=0)
    self.pool = nn.MaxPool2d(kernel_size=(module_H, module_W), stride=(module_H, module_W))

    self.classifier = build_classifier(
      module_C=stem_module_dim, module_H=None, module_W=None, num_answers=len(vocab['answer_token_to_idx']),
      fc_dims=fc_dims, proj_dim=None, downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )

  def forward(self, questions, feats):
    feats = self.stem(feats)
    N, C, H, W = feats.size()
    feats = self.conv(feats)
    feats = self.pool(feats)
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
               use_coords=None, film=False, cl_kernel_size=1, cl_early_fusion=False,
               relational_module=False, rel_image_dim=24, rel_module_dim=256, rel_num_layers=4,
               multimodal_core=False, mc_module_dim=256, mc_num_layers=4, mc_batchnorm=True, mc_kernel_size=1,
               fc_dims=(1024,), fc_use_batchnorm=False, fc_dropout=0):
    super(CnnLstmModel, self).__init__()
    if film:
      if relational_module:
        rnn_dim = 2 * stem_module_dim + 8 * use_coords
      elif multimodal_core:
        rnn_dim = 2 * stem_module_dim + 4 * use_coords
      else:
        rnn_dim = 2 * stem_module_dim
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

    if use_coords == 1 or (use_coords is None and relational_module):
      use_coords = 1
      self.coords = coord_map((module_H, module_W))
    else:
      use_coords = 0
      self.coords = None

    if film:
      self.film = FiLM()
    else:
      self.film = None

    assert not relational_module or not multimodal_core
    self.relational_module = relational_module
    self.multimodal_core = multimodal_core

    if self.relational_module:
      # https://arxiv.org/abs/1706.01427
      self.conv = nn.Conv2d(stem_module_dim, rel_image_dim, kernel_size=1, padding=0)
      if film:
        self.rel = build_relational_module(
          feature_dim=((rel_image_dim + 2 * use_coords) * 2), module_dim=rel_module_dim,
          num_layers=rel_num_layers
        )
      else:
        self.rel = build_relational_module(
          feature_dim=((rel_image_dim + 2 * use_coords) * 2 + rnn_dim), module_dim=rel_module_dim,
          num_layers=rel_num_layers
        )
      module_C = rel_module_dim

    elif self.multimodal_core:
      # https://arxiv.org/abs/1809.04482
      if film:
        self.mc = build_multimodal_core(
          feature_dim=(stem_module_dim + 2 * use_coords), module_dim=mc_module_dim, num_layers=mc_num_layers,
          with_batchnorm=mc_batchnorm, kernel_size=mc_kernel_size
        )
      else:
        self.mc = build_multimodal_core(
          feature_dim=(stem_module_dim + rnn_dim + 2 * use_coords), module_dim=mc_module_dim, num_layers=mc_num_layers,
          with_batchnorm=mc_batchnorm, kernel_size=mc_kernel_size
        )
      module_C = mc_module_dim

    else:
      self.early_fusion = cl_early_fusion
      if cl_early_fusion and not film:
        self.conv = nn.Conv2d(
          stem_module_dim + 2 * use_coords + rnn_dim, stem_module_dim, kernel_size=cl_kernel_size, padding=cl_kernel_size // 2
        )
        module_C = stem_module_dim
      else:
        self.conv = nn.Conv2d(
          stem_module_dim + 2 * use_coords, stem_module_dim, kernel_size=cl_kernel_size, padding=cl_kernel_size // 2
        )
        if cl_early_fusion or film:
          module_C = stem_module_dim
        else:
          module_C = stem_module_dim + rnn_dim
      self.pool = nn.MaxPool2d(kernel_size=(module_H, module_W), stride=(module_H, module_W))

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
      feats = self.conv(feats)
      if self.coords is not None:
        feats = torch.cat([feats, self.coords.unsqueeze(0).repeat(N, 1, 1, 1)], 1)
      N, C, H, W = feats.size()
      feats = feats.view(N, C, H * W).permute(0, 2, 1)
      feats1 = feats.unsqueeze(1)
      feats1 = feats1.repeat(1, H * W, 1, 1)
      if self.film is None:
        feats2 = torch.cat([feats, q_feats.unsqueeze(1).repeat(1, H * W, 1)], 2)
      else:
        feats2 = feats
      feats2 = feats2.unsqueeze(2)
      feats2 = feats2.repeat(1, 1, H * W, 1)
      feats = torch.cat([feats1, feats2], 3)
      if self.film is None:
        feats = feats.view(N * H * W * H * W, C * 2 + Q)
      else:
        feats = feats.permute(0, 3, 1, 2)
        feats = self.film(x=feats, gammas=q_feats[:, :Q // 2], betas=q_feats[:, Q // 2:])
        feats = feats.permute(0, 2, 3, 1).contiguous().view(N * H * W * H * W, C * 2)
      feats = self.rel(feats)
      feats = feats.view(N, H * W * H * W, feats.size(1))
      feats = feats.sum(1)

    elif self.multimodal_core:
      if self.coords is not None:
        feats = torch.cat([feats, self.coords.unsqueeze(0).repeat(N, 1, 1, 1)], 1)
      if self.film is None:
        q_feats = q_feats.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        feats = torch.cat([feats, q_feats], 1)
      else:
        feats = self.film(x=feats, gammas=q_feats[:, :Q // 2], betas=q_feats[:, Q // 2:])
      feats = self.mc(feats)
      feats = feats.sum(3).sum(2)

    else:
      if self.coords is not None:
        feats = torch.cat([feats, self.coords.unsqueeze(0).repeat(N, 1, 1, 1)], 1)
      if self.early_fusion:
        if self.film is None:
          q_feats = q_feats.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
          feats = torch.cat([feats, q_feats], 1)
        else:
          feats = self.film(x=feats, gammas=q_feats[:, :Q // 2], betas=q_feats[:, Q // 2:])
      feats = self.conv(feats)
      feats = self.pool(feats)
      if self.early_fusion:
        feats = feats.view(N, -1)
      else:
        if self.film is None:
          feats = feats.view(N, -1)
          feats = torch.cat([q_feats, feats], 1)
        else:
          feats = self.film(x=feats, gammas=q_feats[:, :Q // 2], betas=q_feats[:, Q // 2:])
          feats = feats.view(N, -1)

    scores = self.classifier(feats)
    return scores


class CnnLstmSaModel(nn.Module):
  def __init__(self, vocab,
               rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
               feature_dim=(1024, 14, 14), stem_module_dim=128,
               stem_use_resnet=False, stem_resnet_fixed=False, resnet_model_stage=3,
               stem_num_layers=2, stem_batchnorm=False, stem_kernel_size=3,
               stem_stride=1, stem_stride2_freq=0, stem_padding=None, use_coords=0, film=False,
               stacked_attn_dim=512, num_stacked_attn=2, sa_kernel_size=1,
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

    if use_coords == 1:
      self.coords = coord_map((module_H, module_W))
    else:
      use_coords = 0
      self.coords = None

    self.image_proj = nn.Conv2d(stem_module_dim, stacked_attn_dim - 2 * use_coords, kernel_size=1, padding=0)
    self.ques_proj = nn.Linear(rnn_dim, stacked_attn_dim)
    self.stacked_attns = []
    for i in range(num_stacked_attn):
      sa = StackedAttention(stacked_attn_dim, stacked_attn_dim, kernel_size=sa_kernel_size, film=film)
      # sa = StackedAttention(rnn_dim, stacked_attn_dim)
      self.stacked_attns.append(sa)
      self.add_module('stacked-attn-%d' % i, sa)

    self.classifier = build_classifier(
      module_C=stacked_attn_dim, module_H=None, module_W=None,
      num_answers=len(vocab['answer_token_to_idx']), fc_dims=fc_dims, proj_dim=None,
      downsample=None, with_batchnorm=fc_use_batchnorm, dropout=fc_dropout
    )
    init_modules(self.modules(), init='normal')

  def forward(self, questions, feats):
    u = self.rnn(questions)
    feats = self.stem(feats)
    feats = self.image_proj(feats)
    N = feats.size()[0]
    if self.coords is not None:
      feats = torch.cat([feats, self.coords.unsqueeze(0).repeat(N, 1, 1, 1)], 1)
    u = self.ques_proj(u)

    for sa in self.stacked_attns:
      u = sa(feats, u)

    scores = self.classifier(u)

    # u1 = self.ques_proj(u)

    # for sa in self.stacked_attns:
    #   u1 = sa(v, u, u1)

    # scores = self.classifier(u1)
    return scores
