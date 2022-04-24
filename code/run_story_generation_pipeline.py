# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging
import argparse
import json
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BartForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: "
                             "allenai/unifiedqa-t5-{small, base, large ...} ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=32, # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--gen_storyline_len",
                        default=128,  # 8 * 8 * 5
                        type=int,
                        help="The maximum length for generated storyline.")
    parser.add_argument("--sub_sample",
                        default=0, # 8 * 8 * 5
                        type=int,
                        help="0 means full data; otherwise, use K random sample for training")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--pos_weight',
                        type=int,
                        default=1,
                        help="positive weight on label 1")
    parser.add_argument("--load_model",
                        type=str,
                        help="pretrained model dir",
                        default="")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument('--save_model',
                        action='store_true',
                        help="save best or not")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    parser.add_argument('--input_event_num',
                        type=int,
                        default=1,
                        help="input event number")
    parser.add_argument('--mu',
                        type=int,
                        default=0,
                        help="exponential decay factor; default 0: not using")
    parser.add_argument('--num_beams',
                        type=int,
                        default=4,
                        help="number of beams; default 4 (bart)")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(torch.device('cuda:%s' % args.device_num))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        logger.info(torch.cuda.current_device())
        n_gpu = len(args.device_num.split(','))
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_relation = True if "with_rel" in args.output_dir else False
    no_struct = True if "no_struct" in args.output_dir else False
    use_story_input = True if "story_input" in args.output_dir else False

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # construct model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if no_struct:
        num_added_toks = tokenizer.add_tokens(['<eoe>'])
    else:
        num_added_toks = tokenizer.add_tokens(['<eoe>', '<before>', '<after>', '<vague>'])

    logger.info('We have added %s tokens' % num_added_toks)

    model = BartForConditionalGeneration.from_pretrained(args.model)
    story_model = BartForConditionalGeneration.from_pretrained(args.model)

    model.resize_token_embeddings(len(tokenizer))
    story_model.resize_token_embeddings(len(tokenizer))

    if args.load_model:
        logger.info(args.load_model)
        model_state_dict = torch.load(args.load_model, map_location=device)
        model.load_state_dict(model_state_dict)

    model.to(device)
    story_model.to(device)
    if args.do_train:
        with open("%s%s%s" % (args.data_dir, "train", args.file_suffix)) as infile:
            train_data = json.load(infile)

        if args.sub_sample > 0:
            random.Random(args.seed).shuffle(train_data)
            train_data = train_data[:args.sub_sample]

        if 'wp' in args.file_suffix:
            train_features = convert_to_wp_features(train_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        else:
            train_features = convert_graph_to_features(train_data, input_event_num=args.input_event_num,
                                                       use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                       use_story_input=use_story_input)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        all_inputs = select_field(train_features, 'inputs')
        encoded_inputs = tokenizer(all_inputs, padding=True, truncation=True, return_tensors="pt")

        all_input_ids = encoded_inputs['input_ids']
        all_input_mask = encoded_inputs['attention_mask']

        all_labels = select_field(train_features, 'labels')
        encoded_outputs = tokenizer(all_labels, padding=True, truncation=True, return_tensors="pt")

        all_output_ids = encoded_outputs['input_ids']
        all_output_mask = encoded_outputs['attention_mask']

        all_stories = select_field(train_features, 'story')
        encoded_outputs_stories = tokenizer(all_stories, padding=True, truncation=True, return_tensors="pt")

        all_output_ids_stories = encoded_outputs_stories['input_ids']
        all_output_mask_stories = encoded_outputs_stories['attention_mask']

        if use_story_input:
            all_story_inputs = select_field(train_features, 'story_input')

        all_key_indices = torch.tensor(list(range(len(all_labels))), dtype=torch.long)

        logger.info("id_size: {}, mask_size: {}, instance_key_size: {}, label_size: {}, story_size: {}".format(
            all_input_ids.size(), all_input_mask.size(), all_key_indices.size(), all_output_ids.size(),
            all_output_ids_stories.size()))

        train_data = TensorDataset(all_input_ids, all_input_mask, all_key_indices,
                                   all_output_ids, all_output_mask, all_output_ids_stories, all_output_mask_stories)

        with open("%s%s%s" % (args.data_dir, "dev", args.file_suffix)) as infile:
            eval_data = json.load(infile)

        if 'wp' in args.file_suffix:
            eval_features = convert_to_wp_features(eval_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        else:
            eval_features = convert_graph_to_features(eval_data, input_event_num=0, is_eval=True,
                                                      use_relation=use_relation, no_struct=no_struct,
                                                      use_story_input=use_story_input)

        # free memory
        del train_features

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        story_model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_to_save_s = model.module if hasattr(story_model, 'module') else story_model
        output_model_file_s = os.path.join(args.output_dir, "pytorch_model_s.bin")
        output_perf_file = os.path.join(args.output_dir, "dev_perf.txt")

        # Prepare optimizer
        param_optimizer = list(model.named_parameters()) + list(story_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = num_train_steps

        if args.fp16:
            try:
                from apex.optimizers import FusedAdam
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        best_eval_perplexity = float('inf')
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc_start, tr_acc_end = 0.0, 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_masks, instance_indices, output_ids, output_masks, output_ids_s, output_masks_s = batch
                if 't5' in args.model:
                    loss, _, _, _ = model(input_ids, attention_mask=input_masks,
                                      labels=output_ids, decoder_attention_mask=output_masks)
                    loss_s = 0.0
                else:
                    output = model(input_ids, attention_mask=input_masks,
                                   labels=output_ids,  decoder_attention_mask=output_masks)
                    loss = output[0]

                    storylines = model.generate(input_ids, attention_mask=input_masks,
                                                max_length=args.gen_storyline_len, num_beams=args.num_beams)

                    p = args.mu / (args.mu + np.exp(global_step / args.mu)) if args.mu != 0 else 0.0
                    if use_story_input:
                        storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer, all_story_inputs,
                                                          instance_indices, thresh=p)
                    else:
                        storylines = storylines[:, 1:]

                    storyline_masks = torch.where(storylines == 1, 0, 1)

                    loss_s = story_model(storylines, attention_mask=storyline_masks,
                                         labels=output_ids_s, decoder_attention_mask=output_masks_s)[0]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    loss_s = loss_s.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_s = loss_s / args.gradient_accumulation_steps

                # combine loss
                loss += loss_s

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += output_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if step > 0 and step % 1000 == 0:
                    logger.info("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                    logger.info("global_step: {}, p: {}".format(global_step, p))

            if args.do_eval:
                # eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                # eval_input_mask = torch.tensor(select_field(eval_features, 'attention_mask'), dtype=torch.long)

                eval_inputs = select_field(eval_features, 'inputs')
                eval_encoded_inputs = tokenizer(eval_inputs, padding=True, truncation=True, return_tensors="pt")

                eval_input_ids = eval_encoded_inputs['input_ids']
                eval_input_mask = eval_encoded_inputs['attention_mask']

                eval_labels = select_field(eval_features, 'labels')
                eval_encoded_outputs = tokenizer(eval_labels, padding=True, truncation=True, return_tensors="pt")
                eval_output_ids = eval_encoded_outputs['input_ids']
                eval_output_mask = eval_encoded_outputs['attention_mask']

                eval_stories = select_field(eval_features, 'story')
                eval_outputs_stories = tokenizer(eval_stories, padding=True, truncation=True, return_tensors="pt")

                eval_output_ids_stories = eval_outputs_stories['input_ids']
                eval_output_ids_stories[eval_output_ids_stories == tokenizer.pad_token_id] = -100
                eval_output_mask_stories = eval_outputs_stories['attention_mask']

                if use_story_input:
                    eval_story_inputs = select_field(eval_features, 'story_input')

                eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

                eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids,
                                          eval_output_mask, eval_output_ids_stories, eval_output_mask_stories)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                story_model.eval()

                perplexity = []
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_masks, instance_indices, output_ids, output_masks, output_ids_s, output_masks_s = batch

                    with torch.no_grad():

                        storylines = model.generate(input_ids, attention_mask=input_masks,
                                                    max_length=args.gen_storyline_len, num_beams=args.num_beams)
                        if use_story_input:
                            storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer,
                                                              eval_story_inputs, instance_indices)
                        else:
                            storylines = storylines[:, 1:]

                        storyline_masks = torch.where(storylines == 1, 0, 1)

                        eval_loss_s = story_model(storylines, attention_mask=storyline_masks,
                                                  labels=output_ids_s, decoder_attention_mask=output_masks_s)[0]
                        perplexity.append(torch.exp(eval_loss_s).item())

                logger.info("Sequence Perplexity %.4f" % np.mean(perplexity))

                if np.mean(perplexity) < best_eval_perplexity:
                    best_eval_perplexity = np.mean(perplexity)
                    logger.info("Save at Epoch %s" % epoch)
                    with open(output_perf_file, 'w') as outfile:
                        outfile.write("%.4f" % best_eval_perplexity)
                    if args.save_model:
                        torch.save(model_to_save.state_dict(), output_model_file)
                        torch.save(model_to_save_s.state_dict(), output_model_file_s)

                model.train()
                story_model.train()

if __name__ == "__main__":
    main()
