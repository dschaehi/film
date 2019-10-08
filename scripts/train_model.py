#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import ipdb as pdb
import json
import random
import shutil
from termcolor import colored
import time

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import vr.utils as utils
from vr.preprocess import decode, encode, tokenize
from vr.programs import list_to_prefix, list_to_str

from vr.data import ClevrDataset, ClevrDataLoader
from vr.models import ModuleNet, Seq2Seq, CnnModel, LstmModel, CnnLstmModel, CnnLstmSaModel
from vr.models import FiLMedNet
from vr.models import FiLMGen

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=None, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# ShapeWorld input data (ignores all of the above)
parser.add_argument('--sw_type', default='agreement')
parser.add_argument('--sw_name', default=None)
parser.add_argument('--sw_variant', default=None)
parser.add_argument('--sw_language', default=None)
parser.add_argument('--sw_config', default=None)
parser.add_argument('--sw_mixer', default=0, type=int)
parser.add_argument('--sw_features', default=0, type=int)
parser.add_argument('--sw_program', default=0, type=int)  # 0: none, 1: pn, 2: rpn, 3: clevr-style

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
  choices=['FiLM', 'PG', 'EE', 'PG+EE', 'CNN', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA', 'FiLM+BoW', 'FiLM+ResNet1', 'FiLM+ResNet0'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net / FiLMedNet options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# FiLM only options
parser.add_argument('--set_execution_engine_eval', default=0, type=int)
parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
# hx: parser.add_argument('--encoder_type', default='gru', type=str,
  # hx: choices=['linear', 'gru', 'lstm'])
parser.add_argument('--encoder_type', default='gru', type=str,
  choices=['linear', 'gru', 'lstm', 'bow'])
parser.add_argument('--decoder_type', default='linear', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp'])
parser.add_argument('--gamma_baseline', default=1, type=float)
parser.add_argument('--num_modules', default=4, type=int)
parser.add_argument('--module_stem_kernel_size', default=3, type=int)
parser.add_argument('--module_stem_stride2_freq', default=0, type=int)
parser.add_argument('--module_stem_padding', default=None, type=int)
parser.add_argument('--module_num_layers', default=1, type=int)  # Only mnl=1 currently implemented
parser.add_argument('--module_batchnorm_affine', default=0, type=int)  # 1 overrides other factors
parser.add_argument('--module_dropout', default=5e-2, type=float)
parser.add_argument('--module_input_proj', default=1, type=int)  # Inp conv kernel size (0 for None)
parser.add_argument('--module_kernel_size', default=3, type=int)
parser.add_argument('--condition_method', default='bn-film', type=str,
  choices=['block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
parser.add_argument('--condition_pattern', default='', type=str)  # List of 0/1's (len = # FiLMs)
parser.add_argument('--use_gamma', default=1, type=int)
parser.add_argument('--use_beta', default=1, type=int)
parser.add_argument('--use_coords', default=1, type=int)  # 0: none, 1: low usage, 2: high usage
parser.add_argument('--film', default=0, type=int)
parser.add_argument('--cl_kernel_size', default=1, type=int)
parser.add_argument('--cl_early_fusion', default=0, type=int)
parser.add_argument('--grad_clip', default=0, type=float)  # <= 0 for no grad clipping
parser.add_argument('--debug_every', default=float('inf'), type=float)  # inf for no pdb
parser.add_argument('--print_verbose_every', default=float('inf'), type=float)  # inf for min print

# Relational Module for CNN+LSTM (https://arxiv.org/abs/1706.01427)
# (code adapted from https://github.com/kimhc6028/relational-networks)
parser.add_argument('--relational_module', default=0, type=int)
parser.add_argument('--rel_image_dim', default=24, type=int)
parser.add_argument('--rel_module_dim', default=256, type=int)
parser.add_argument('--rel_num_layers', default=4, type=int)

# Multimodal Core options for CNN+LSTM (https://arxiv.org/abs/1809.04482)
parser.add_argument('--multimodal_core', default=0, type=int)
parser.add_argument('--mc_module_dim', default=256, type=int)
parser.add_argument('--mc_num_layers', default=4, type=int)
parser.add_argument('--mc_batchnorm', default=1, type=int)
parser.add_argument('--mc_kernel_size', default=1, type=int)

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)
parser.add_argument('--sa_kernel_size', default=1, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
  choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
           'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default='0')

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--optimizer', default='Adam',
  choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='checkpoint.ckpt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
parser.add_argument('--record_loss_every', default=100, type=int)
parser.add_argument('--record_accuracy_10k_every', default=1000, type=int)
parser.add_argument('--record_accuracy_every', default=5000, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--time', default=0, type=int)


def main(args):
  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)
  print('Will save checkpoints to %s' % args.checkpoint_path)

  if args.sw_name is not None or args.sw_config is not None:
    from shapeworld import Dataset, torch_util
    from shapeworld.datasets import clevr_util

    class ShapeWorldDataLoader(torch_util.ShapeWorldDataLoader):

      def __iter__(self):
        for batch in super(ShapeWorldDataLoader, self).__iter__():
          if 'caption' in batch:
            question = batch['caption'].long()
          else:
            question = batch['question'].long()
          if args.sw_features == 1:
            image = batch['world_features']
          else:
            image = batch['world']
          feats = image
          if 'agreement' in batch:
            answer = batch['agreement'].long()
          else:
            answer = batch['answer'].long()
          if 'caption_model' in batch:
            assert args.sw_name.startswith('clevr') or args.sw_program == 3
            program_seq = batch['caption_model']
            # .apply_(callable=(lambda model: clevr_util.parse_program(mode=0, model=model)))
          elif 'question_model' in batch:
            program_seq = batch['question_model']
          elif 'caption' in batch:
            if args.sw_program == 1:
              program_seq = batch['caption_pn'].long()
            elif args.sw_program == 2:
              program_seq = batch['caption_rpn'].long()
            else:
              program_seq = [None]
          else:
            program_seq = [None]
          # program_seq = torch.IntTensor([0 for _ in batch['question']])
          program_json = dict()
          yield question, image, feats, answer, program_seq, program_json

    # exclude_values = ('world',) if args.sw_features == 1 else ('world_features',)
    # dataset = Dataset.create(dtype=args.sw_type, name=args.sw_name, variant=args.sw_variant,
    #   language=args.sw_language, config=args.sw_config, exclude_values=exclude_values)
    dataset = Dataset.create(dtype=args.sw_type, name=args.sw_name, variant=args.sw_variant,
      language=args.sw_language, config=args.sw_config)
    print('ShapeWorld dataset: {} (variant: {})'.format(dataset, args.sw_variant))
    print('Config: ' + str(args.sw_config))

    if args.program_generator_start_from is None and args.execution_engine_start_from is None and args.baseline_start_from is None:
      if args.sw_name.startswith('clevr'):
        assert args.sw_program in (0, 3)
        # from vocab.json
        question_token_to_idx = {
          word: index + 2 for word, index in dataset.vocabularies['language'].items() if index > 0
        }
        question_token_to_idx['<NULL>'] = 0
        question_token_to_idx['<START>'] = 1
        question_token_to_idx['<END>'] = 2
        program_token_to_idx = {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "count": 4, "equal_color": 5, "equal_integer": 6, "equal_material": 7, "equal_shape": 8, "equal_size": 9, "exist": 10, "filter_color[blue]": 11, "filter_color[brown]": 12, "filter_color[cyan]": 13, "filter_color[gray]": 14, "filter_color[green]": 15, "filter_color[purple]": 16, "filter_color[red]": 17, "filter_color[yellow]": 18, "filter_material[metal]": 19, "filter_material[rubber]": 20, "filter_shape[cube]": 21, "filter_shape[cylinder]": 22, "filter_shape[sphere]": 23, "filter_size[large]": 24, "filter_size[small]": 25, "greater_than": 26, "intersect": 27, "less_than": 28, "query_color": 29, "query_material": 30, "query_shape": 31, "query_size": 32, "relate[behind]": 33, "relate[front]": 34, "relate[left]": 35, "relate[right]": 36, "same_color": 37, "same_material": 38, "same_shape": 39, "same_size": 40, "scene": 41, "union": 42, "unique": 43}
        answer_token_to_idx = {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "0": 4, "1": 5, "10": 6, "2": 7, "3": 8, "4": 9, "5": 10, "6": 11, "7": 12, "8": 13, "9": 14, "blue": 15, "brown": 16, "cube": 17, "cyan": 18, "cylinder": 19, "gray": 20, "green": 21, "large": 22, "metal": 23, "no": 24, "purple": 25, "red": 26, "rubber": 27, "small": 28, "sphere": 29, "yellow": 30, "yes": 31}
        vocab = dict(
          question_token_to_idx=question_token_to_idx,
          program_token_to_idx=program_token_to_idx,
          answer_token_to_idx=answer_token_to_idx
        )
      else:
        question_token_to_idx = {
          word: index + 2 for word, index in dataset.vocabularies['language'].items() if index > 0
        }
        question_token_to_idx['<NULL>'] = 0
        question_token_to_idx['<START>'] = 1
        question_token_to_idx['<END>'] = 2
        if args.sw_program == 3:
          program_token_to_idx = {'<NULL>': 0, '<START>': 1, '<END>': 2, 'scene': 3, 'relation[type]': 4, 'quantifier[existential]': 5, 'attribute[shape-square]': 6, 'attribute[shape-rectangle]': 7, 'attribute[shape-triangle]': 8, 'attribute[shape-pentagon]': 9, 'attribute[shape-cross]': 10, 'attribute[shape-circle]': 11, 'attribute[shape-semicircle]': 12, 'attribute[shape-ellipse]': 13, 'attribute[color-red]': 14, 'attribute[color-green]': 15, 'attribute[color-blue]': 16, 'attribute[color-yellow]': 17, 'attribute[color-magenta]': 18, 'attribute[color-cyan]': 19, 'attribute[color-gray]': 20, 'relation[attribute]': 21}
          program_token_num_inputs = {'<NULL>': 1, '<START>': 1, '<END>': 1, 'scene': 0, 'relation[type]': 1, 'quantifier[existential]': 2, 'attribute[shape-square]': 1, 'attribute[shape-rectangle]': 1, 'attribute[shape-triangle]': 1, 'attribute[shape-pentagon]': 1, 'attribute[shape-cross]': 1, 'attribute[shape-circle]': 1, 'attribute[shape-semicircle]': 1, 'attribute[shape-ellipse]': 1, 'attribute[color-red]': 1, 'attribute[color-green]': 1, 'attribute[color-blue]': 1, 'attribute[color-yellow]': 1, 'attribute[color-magenta]': 1, 'attribute[color-cyan]': 1, 'attribute[color-gray]': 1, 'relation[attribute]': 1}
        else:
          program_token_to_idx = {
            word: index + 2 for word, index in dataset.vocabularies['pn'].items() if index > 0
          }
          program_token_to_idx['<NULL>'] = 0
          program_token_to_idx['<START>'] = 1
          program_token_to_idx['<END>'] = 2
          program_token_num_inputs = dict(dataset.pn_arity)
          program_token_num_inputs.pop('')
          program_token_num_inputs['<NULL>'] = 1
          program_token_num_inputs['<START>'] = 1
          program_token_num_inputs['<END>'] = 1
        vocab = dict(
          question_token_to_idx=question_token_to_idx,
          program_token_to_idx=program_token_to_idx,
          program_token_num_inputs=program_token_num_inputs,
          answer_token_to_idx={'false': 0, 'true': 1}
        )
      with open(args.checkpoint_path + '.vocab', 'w') as filehandle:
        json.dump(vocab, filehandle)

    else:
      if args.program_generator_start_from is not None:
        with open(args.program_generator_start_from + '.vocab', 'r') as filehandle:
          vocab = json.load(filehandle)
      elif args.execution_engine_start_from is not None:
        with open(args.execution_engine_start_from + '.vocab', 'r') as filehandle:
          vocab = json.load(filehandle)
      elif args.baseline_start_from is not None:
        with open(args.baseline_start_from + '.vocab', 'r') as filehandle:
          vocab = json.load(filehandle)
      question_token_to_idx = vocab['question_token_to_idx']
      program_token_to_idx = vocab['program_token_to_idx']
      # program_token_num_inputs = vocab['program_token_num_inputs']
      answer_token_to_idx = vocab['answer_token_to_idx']
      index = len(question_token_to_idx)
      for word in dataset.vocabularies['language']:
        if word not in question_token_to_idx:
          question_token_to_idx[word] = index
          index += 1
      with open(args.checkpoint_path + '.vocab', 'w') as filehandle:
        json.dump(vocab, filehandle)

    if args.sw_features == 1:
      shape = dataset.vector_shape(value_name='world_features')
    else:
      shape = dataset.world_shape()
    args.feature_dim = '{},{},{}'.format(shape[2], shape[0], shape[1])
    args.vocab_json = args.checkpoint_path + '.vocab'

    include_model = args.model_type in ('PG', 'EE', 'PG+EE') and (args.sw_name.startswith('clevr') or args.sw_program == 3)
    if include_model:

      def preprocess(model):
        if args.sw_name.startswith('clevr'):
          program_prefix = list_to_prefix(model['program'])
        else:
          program_prefix = clevr_util.parse_program(mode=0, model=model)
        program_str = list_to_str(program_prefix)
        program_tokens = tokenize(program_str)
        program_encoded = encode(program_tokens, program_token_to_idx)
        program_encoded += [program_token_to_idx['<NULL>'] for _ in range(27 - len(program_encoded))]
        return np.asarray(program_encoded, dtype=np.int64)

      if args.sw_name.startswith('clevr'):
        preprocessing = dict(question_model=preprocess)
      else:
        preprocessing = dict(caption_model=preprocess)

    elif args.sw_program in (1, 2):

      def preprocess(caption_pn):
        caption_pn += (caption_pn > 0) * 2
        for n, symbol in enumerate(caption_pn):
          if symbol == 0:
            caption_pn[n] = 2
            break
        caption_pn = np.concatenate(([1], caption_pn))
        return caption_pn

      if args.sw_program == 1:
        preprocessing = dict(caption_pn=preprocess)
      else:
        preprocessing = dict(caption_rpn=preprocess)

    else:
      preprocessing = None

    train_dataset = torch_util.ShapeWorldDataset(
      dataset=dataset, mode='train', include_model=include_model, preprocessing=preprocessing
    )
    train_loader = ShapeWorldDataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.loader_num_workers)

    if args.sw_mixer == 1:
      val_loader = list()
      for d in dataset.datasets:
        val_dataset = torch_util.ShapeWorldDataset(
          dataset=d, mode='validation', include_model=include_model, epoch=(args.num_val_samples is None),
          preprocessing=preprocessing
        )
        val_loader.append(ShapeWorldDataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.loader_num_workers))
    else:
      val_dataset = torch_util.ShapeWorldDataset(
        dataset=dataset, mode='validation', include_model=include_model, epoch=(args.num_val_samples is None),
        preprocessing=preprocessing
      )
      val_loader = ShapeWorldDataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.loader_num_workers)

    train_loop(args, train_loader, val_loader)

  else:
    vocab = utils.load_vocab(args.vocab_json)

    if args.use_local_copies == 1:
      shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
      shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
      shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
      shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
      args.train_question_h5 = '/tmp/train_questions.h5'
      args.train_features_h5 = '/tmp/train_features.h5'
      args.val_question_h5 = '/tmp/val_questions.h5'
      args.val_features_h5 = '/tmp/val_features.h5'

    question_families = None
    if args.family_split_file is not None:
      with open(args.family_split_file, 'r') as f:
        question_families = json.load(f)

    train_loader_kwargs = {
      'question_h5': args.train_question_h5,
      'feature_h5': args.train_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
      'shuffle': args.shuffle_train_data == 1,
      'question_families': question_families,
      'max_samples': args.num_train_samples,
      'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
      'question_h5': args.val_question_h5,
      'feature_h5': args.val_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
      'question_families': question_families,
      'max_samples': args.num_val_samples,
      'num_workers': args.loader_num_workers,
    }

    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
         ClevrDataLoader(**val_loader_kwargs) as val_loader:
      train_loop(args, train_loader, val_loader)

    if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
      os.remove('/tmp/train_questions.h5')
      os.remove('/tmp/train_features.h5')
      os.remove('/tmp/val_questions.h5')
      os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader):
  vocab = utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  # Set up model
  optim_method = getattr(torch.optim, args.optimizer)
  if args.model_type in ['FiLM', 'PG', 'PG+EE', 'FiLM+BoW', 'FiLM+ResNet1', 'FiLM+ResNet0']:
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = optim_method(program_generator.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    print('Here is the conditioning network:')
    print(program_generator)
  if args.model_type in ['FiLM', 'EE', 'PG+EE', 'FiLM+BoW', 'FiLM+ResNet1', 'FiLM+ResNet0']:
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = optim_method(filter(lambda p: p.requires_grad, execution_engine.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    print('Here is the conditioned network:')
    print(execution_engine)
  if args.model_type in ['CNN', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = optim_method(params,
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type
  if torch.cuda.is_available():
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
  else:
    loss_fn = torch.nn.CrossEntropyLoss().cpu()

  if args.sw_name is None and args.sw_config is None:
    stats = {
      'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
      'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
      'best_val_acc': -1, 'model_t': 0,
    }
  else:
    stats = {
      'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
      'val_accs': [], 'val_accs_ts': [], 'best_val_acc': -1, 'model_t': 0,
    }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [program_generator, execution_engine, baseline_model])

  if args.sw_name is None and args.sw_config is None:
    print('train_loader has %d samples' % len(train_loader.dataset))
    print('val_loader has %d samples' % len(val_loader.dataset))

  num_checkpoints = 0
  epoch_start_time = 0.0
  epoch_total_time = 0.0
  train_pass_total_time = 0.0
  val_pass_total_time = 0.0
  running_loss = 0.0
  while t < args.num_iterations:
    if (epoch > 0) and (args.time == 1):
      epoch_time = time.time() - epoch_start_time
      epoch_total_time += epoch_time
      print(colored('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white'))
      print(colored('Epoch Pass Time      : ' + str(epoch_time), 'white'))
    epoch_start_time = time.time()

    epoch += 1
    print('Starting epoch %d' % epoch)
    train_loader_iter = iter(train_loader)
    while True:
      try:
        batch = next(train_loader_iter)
      except StopIteration:
        break
      t += 1
      if (args.sw_name is None and args.sw_config is None) or args.sw_features == 1:
        questions, _, feats, answers, programs, _ = batch
      else:
        questions, feats, _, answers, programs, _ = batch
      if isinstance(questions, list):
        questions = questions[0]
      if torch.cuda.is_available():
        questions_var = Variable(questions.cuda())
        feats_var = Variable(feats.cuda())
        answers_var = Variable(answers.cuda())
        if programs[0] is not None:
          programs_var = Variable(programs.cuda())
      else:
        questions_var = Variable(questions.cpu())
        feats_var = Variable(feats.cpu())
        answers_var = Variable(answers.cpu())
        if programs[0] is not None:
          programs_var = Variable(programs.cpu())

      reward = None
      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      elif args.model_type in ['CNN', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          if torch.cuda.is_available():
            program_generator.reinforce_backward(centered_reward.cuda())
          else:
            program_generator.reinforce_backward(centered_reward.cpu())
          pg_optimizer.step()
      elif args.model_type.startswith('FiLM'):
        if args.set_execution_engine_eval == 1:
          set_mode('eval', [execution_engine])
        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred)
        loss = loss_fn(scores, answers_var)

        pg_optimizer.zero_grad()
        ee_optimizer.zero_grad()
        if args.debug_every <= -2:
          pdb.set_trace()
        loss.backward()
        if args.debug_every < float('inf'):
          check_grad_num_nans(execution_engine, 'FiLMedNet')
          check_grad_num_nans(program_generator, 'FiLMGen')

        if args.train_program_generator == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
          pg_optimizer.step()
        if args.train_execution_engine == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
          ee_optimizer.step()

      if t % args.record_loss_every == 0:
        running_loss += loss.data[0]
        avg_loss = running_loss / args.record_loss_every
        print(t, avg_loss, flush=True)
        stats['train_losses'].append(avg_loss)
        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)
        running_loss = 0.0
      else:
        running_loss += loss.data[0]

      if (t <= 10000 and t % args.record_accuracy_10k_every == 0) \
          or t % args.record_accuracy_every == 0 \
          or t == 1 or t == args.num_iterations:

        if args.sw_name is None and args.sw_config is None:
          print('Checking training accuracy ... ')
          start = time.time()
          train_acc = check_accuracy(args, program_generator, execution_engine,
                                     baseline_model, train_loader)
          if args.time == 1:
            train_pass_time = (time.time() - start)
            train_pass_total_time += train_pass_time
            print(colored('TRAIN PASS AVG TIME: ' + str(train_pass_total_time / num_checkpoints), 'red'))
            print(colored('Train Pass Time      : ' + str(train_pass_time), 'red'))
          print('train accuracy is', train_acc, flush=True)
          stats['train_accs'].append(train_acc)

        print('Checking validation accuracy ...')
        start = time.time()
        val_acc = check_accuracy(args, program_generator, execution_engine,
                                 baseline_model, val_loader)
        if args.time == 1:
          val_pass_time = (time.time() - start)
          val_pass_total_time += val_pass_time
          print(colored('VAL PASS AVG TIME:   ' + str(val_pass_total_time / num_checkpoints), 'cyan'))
          print(colored('Val Pass Time        : ' + str(val_pass_time), 'cyan'))
        if isinstance(val_acc, list):
          for loader, acc in zip(val_loader, val_acc):
            print('val accuracy for', loader.dataset.dataset, 'is', acc)
          val_acc = sum(val_acc) / len(val_acc)
        print('val accuracy is', val_acc, flush=True)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        if val_acc > stats['best_val_acc']:
          stats['best_val_acc'] = val_acc
          stats['model_t'] = t
          best_pg_state = get_state(program_generator)
          best_ee_state = get_state(execution_engine)
          best_baseline_state = get_state(baseline_model)

      if t % args.checkpoint_every == 0 or t == args.num_iterations:
        num_checkpoints += 1
        checkpoint = {
          'args': args.__dict__,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'baseline_kwargs': baseline_kwargs,
          'baseline_state': best_baseline_state,
          'baseline_type': baseline_type,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to', args.checkpoint_path, flush=True)
        torch.save(checkpoint, args.checkpoint_path)
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        with open(args.checkpoint_path + '.json', 'w') as f:
          json.dump(checkpoint, f)

      if t == args.num_iterations:
        break


def parse_int_list(s):
  if s == '': return ()
  return tuple(int(n) for n in s.split(','))


def parse_float_list(s):
  if s == '': return ()
  return tuple(float(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(
      args.program_generator_start_from, model_type=args.model_type)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['question_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    if args.model_type.startswith('FiLM'):
      kwargs['parameter_efficient'] = args.program_generator_parameter_efficient == 1
      kwargs['output_batchnorm'] = args.rnn_output_batchnorm == 1
      kwargs['bidirectional'] = args.bidirectional == 1
      kwargs['encoder_type'] = args.encoder_type
      kwargs['decoder_type'] = args.decoder_type
      kwargs['gamma_option'] = args.gamma_option
      kwargs['gamma_baseline'] = args.gamma_baseline
      kwargs['num_modules'] = args.num_modules
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_dim'] = args.module_dim
      kwargs['debug_every'] = args.debug_every
      if args.model_type == 'FiLM+BoW':
        kwargs['encoder_type'] = 'bow'
      pg = FiLMGen(**kwargs)
    else:
      pg = Seq2Seq(**kwargs)
  if torch.cuda.is_available():
    pg.cuda()
  else:
    pg.cpu()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = utils.load_execution_engine(
      args.execution_engine_start_from, model_type=args.model_type)
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_use_resnet': (args.model_type == 'FiLM+ResNet1' or args.model_type == 'FiLM+ResNet0'),
      'stem_resnet_fixed': args.model_type == 'FiLM+ResNet0',
      'stem_num_layers': args.module_stem_num_layers,
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_kernel_size': args.module_stem_kernel_size,
      'stem_stride2_freq': args.module_stem_stride2_freq,
      'stem_padding': args.module_stem_padding,
      'module_dim': args.module_dim,
      'module_residual': args.module_residual == 1,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': parse_float_list(args.classifier_dropout),
    }
    if args.model_type.startswith('FiLM'):
      kwargs['num_modules'] = args.num_modules
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
      kwargs['module_dropout'] = args.module_dropout
      kwargs['module_input_proj'] = args.module_input_proj
      kwargs['module_kernel_size'] = args.module_kernel_size
      kwargs['use_gamma'] = args.use_gamma == 1
      kwargs['use_beta'] = args.use_beta == 1
      kwargs['use_coords'] = args.use_coords
      kwargs['debug_every'] = args.debug_every
      kwargs['print_verbose_every'] = args.print_verbose_every
      kwargs['condition_method'] = args.condition_method
      kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
      ee = FiLMedNet(**kwargs)
    else:
      ee = ModuleNet(**kwargs)
  if torch.cuda.is_available():
    ee.cuda()
  else:
    ee.cpu()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'CNN':
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_module_dim': args.module_dim,
      'stem_use_resnet': (args.model_type == 'FiLM+ResNet1' or args.model_type == 'FiLM+ResNet0'),
      'stem_resnet_fixed': args.model_type == 'FiLM+ResNet0',
      'stem_num_layers': args.module_stem_num_layers,
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_kernel_size': args.module_stem_kernel_size,
      'stem_stride2_freq': args.module_stem_stride2_freq,
      'stem_padding': args.module_stem_padding,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': parse_float_list(args.classifier_dropout),
    }
    model = CnnModel(**kwargs)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': parse_float_list(args.classifier_dropout),
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_module_dim': args.module_dim,
      'stem_use_resnet': (args.model_type == 'FiLM+ResNet1' or args.model_type == 'FiLM+ResNet0'),
      'stem_resnet_fixed': args.model_type == 'FiLM+ResNet0',
      'stem_num_layers': args.module_stem_num_layers,
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_kernel_size': args.module_stem_kernel_size,
      'stem_stride2_freq': args.module_stem_stride2_freq,
      'stem_padding': args.module_stem_padding,
      'use_coords': args.use_coords,
      'film': args.film == 1,
      'cl_kernel_size': args.cl_kernel_size,
      'cl_early_fusion': args.cl_early_fusion == 1,
      # https://arxiv.org/abs/1706.01427
      'relational_module': args.relational_module == 1,
      'rel_image_dim': args.rel_image_dim,
      'rel_module_dim': args.rel_module_dim,
      'rel_num_layers': args.rel_num_layers,
      # https://arxiv.org/abs/1809.04482
      'multimodal_core': args.multimodal_core == 1,
      'mc_module_dim': args.mc_module_dim,
      'mc_num_layers': args.mc_num_layers,
      'mc_batchnorm': args.mc_batchnorm == 1,
      'mc_kernel_size': args.mc_kernel_size,
      # Default CNN+LSTM
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': parse_float_list(args.classifier_dropout),
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_module_dim': args.module_dim,
      'stem_use_resnet': (args.model_type == 'FiLM+ResNet1' or args.model_type == 'FiLM+ResNet0'),
      'stem_resnet_fixed': args.model_type == 'FiLM+ResNet0',
      'stem_num_layers': args.module_stem_num_layers,
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_kernel_size': args.module_stem_kernel_size,
      'stem_stride2_freq': args.module_stem_stride2_freq,
      'stem_padding': args.module_stem_padding,
      'use_coords': args.use_coords,
      'film': args.film == 1,
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'sa_kernel_size': args.sa_kernel_size,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': parse_float_list(args.classifier_dropout),
    }
    model = CnnLstmSaModel(**kwargs)
  if args.model_type != 'CNN' and model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  if torch.cuda.is_available():
    model.cuda()
  else:
    model.cpu()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
  if isinstance(loader, list):
    accuracies = list()
    for l in loader:
      accuracies.append(check_accuracy(args, program_generator, execution_engine, baseline_model, l))
    return accuracies

  set_mode('eval', [program_generator, execution_engine, baseline_model])
  num_correct, num_samples = 0, 0
  for batch in loader:
    if (args.sw_name is None and args.sw_config is None) or args.sw_features == 1:
      questions, _, feats, answers, programs, _ = batch
    else:
      questions, feats, _, answers, programs, _ = batch

    if isinstance(questions, list):
      questions = questions[0]

    if torch.cuda.is_available():
      questions_var = Variable(questions.cuda(), volatile=True)
      feats_var = Variable(feats.cuda(), volatile=True)
      answers_var = Variable(feats.cuda(), volatile=True)
      if programs[0] is not None:
        programs_var = Variable(programs.cuda(), volatile=True)
    else:
      questions_var = Variable(questions.cpu(), volatile=True)
      feats_var = Variable(feats.cpu(), volatile=True)
      answers_var = Variable(feats.cpu(), volatile=True)
      if programs[0] is not None:
        programs_var = Variable(programs.cpu(), volatile=True)

    scores = None  # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        if torch.cuda.is_available():
          program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(), volatile=True))
        else:
          program_pred = program_generator.sample(Variable(questions[i:i+1].cpu(), volatile=True))
        program_pred_str = decode(program_pred, vocab['program_idx_to_token'])
        program_str = decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
      scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type.startswith('FiLM'):
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['CNN', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      # print(preds)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if args.num_val_samples is not None and num_samples >= args.num_val_samples:
      break

  set_mode('train', [program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / num_samples
  return acc

def check_grad_num_nans(model, model_name='model'):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    num_nans = [np.sum(np.isnan(grad.data.cpu().numpy())) for grad in grads]
    nan_checks = [num_nan == 0 for num_nan in num_nans]
    if False in nan_checks:
      print('Nans in ' + model_name + ' gradient!')
      print(num_nans)
      pdb.set_trace()
      raise(Exception)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
