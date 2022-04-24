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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BartForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path
import json
from nlgeval import NLGEval

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
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=320, # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--gen_storyline_len",
                        default=128,  # 8 * 8 * 5
                        type=int,
                        help="The maximum length for generated storyline.")
    parser.add_argument("--model_dir",
                        type=str,
                        help="saved model dir",
                        default="")
    parser.add_argument("--topk_sample",
                        action='store_true')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
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
    parser.add_argument("--no_struct",
                        action='store_true',
                        help="Whether not to use structure in the input")
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
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    parser.add_argument('--input_event_num',
                        type=int,
                        default=1,
                        help="input event number")
    parser.add_argument('--no_label',
                        action='store_true',
                        help="predict unlabeled data")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(torch.device('cuda:%s' % args.device_num))

    use_relation = True if "with_rel" in args.model_dir else False
    no_struct = True if "no_struct" in args.model_dir else False
    use_story_input = True if "story_input" in args.model_dir else False

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

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # construct model
    logger.info(args.model_dir)
    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin", map_location=device)
    model_state_dict_s = torch.load(args.model_dir + "pytorch_model_s.bin", map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, state_dict=model_state_dict)
    if no_struct:
        num_added_toks = tokenizer.add_tokens(['<eoe>'])
    else:
        num_added_toks = tokenizer.add_tokens(['<eoe>', '<before>', '<after>', '<vague>'])
    logger.info('We have added %s tokens' % num_added_toks)

    model = BartForConditionalGeneration.from_pretrained(args.model)
    story_model = BartForConditionalGeneration.from_pretrained(args.model)

    model.resize_token_embeddings(len(tokenizer))
    story_model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(model_state_dict)
    story_model.load_state_dict(model_state_dict_s)

    model.to(device)
    story_model.to(device)
    del model_state_dict
    del model_state_dict_s

    # Prepare optimizer
    param_optimizer = list(model.named_parameters()) + list(story_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

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
                             t_total=1)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split in ['test']:
        with open("%s%s%s" % (args.data_dir, split, args.file_suffix)) as infile:
            eval_data = json.load(infile)

        if 'wp' in args.file_suffix:
            eval_features = convert_to_wp_features(eval_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        else:
            eval_features = convert_graph_to_features(eval_data, input_event_num=0, use_relation=use_relation,
                                                      no_struct=no_struct, use_story_input=True, is_eval=True)

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

        eval_story_inputs = select_field(eval_features, 'story_input')

        eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

        logger.info("id_size: {}, mask_size: {}, instance_key_size: {}, label_size: {}".format(
            eval_input_ids.size(), eval_input_mask.size(), eval_key_indices.size(), eval_output_ids.size()))

        data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids,
                             eval_output_mask, eval_output_ids_stories, eval_output_mask_stories)

        # Run prediction for full data
        eval_sampler = SequentialSampler(data)
        eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        preds, golds, events, perplexity = [], [], [], []
        pred_to_eval, gold_to_eval = [], []
        model.eval()
        contexts = []

        wrong_structure, wrong_gold = 0, 0

        perplexity, gen_perplexity, overlaps, overlap_ts, overlap_as = [], [], [], [], []
        repeat, distinct, repeat_f, distinct_f = [[] for _ in range(4)], [[] for _ in range(4)], \
                                                 [[] for _ in range(4)], [[] for _ in range(4)]

        gen_all_perp, gen_avg_perp, gen_perplexity, token_len = [], [], [], []
        pred_storylines = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, instance_indices, output_ids, output_masks, output_ids_s, output_masks_s = batch

            if 'wp' not in args.file_suffix:
                output_ids_s = nullify_story_inputs(output_ids_s, tokenizer, eval_story_inputs, instance_indices)

            with torch.no_grad():

                storylines = model.generate(input_ids, attention_mask=input_masks, max_length=args.gen_storyline_len)
                pred_storylines.extend(tokenizer.batch_decode(storylines, skip_special_tokens=True))

                if use_story_input:
                    storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer,
                                                      eval_story_inputs, instance_indices)
                else:
                    storylines = storylines[:, 1:]

                storyline_masks = torch.where(storylines == 1, 0, 1)

                eval_loss_s = story_model(storylines, attention_mask=storyline_masks,
                                          labels=output_ids_s, decoder_attention_mask=output_masks_s)[0]

                perplexity.append(torch.exp(eval_loss_s).item())

                if args.topk_sample:
                    res = story_model.generate(storylines, attention_mask=storyline_masks, max_length=1024,
                                               return_dict_in_generate=True, output_scores=True,
                                               do_sample=True)
                else:
                    res = story_model.generate(storylines, attention_mask=storyline_masks, max_length=1024,
                                               return_dict_in_generate=True, output_scores=True)

                batch_preds = tokenizer.batch_decode(res.sequences, skip_special_tokens=True)
                batch_golds = [eval_stories[x] for x in instance_indices.tolist()]
                batch_context = [eval_inputs[x] for x in instance_indices.tolist()]

                for i, (pred, gold, context) in enumerate(zip(batch_preds, batch_golds, batch_context)):
                    token_len.append(len(pred.split(' ')))

                    sents = sep_output_sents(pred)
                    gold = sep_output_sents(gold)

                    if 'wp' in args.file_suffix:
                        pred_to_eval.append(pred)
                        start = 0
                    else:
                        pred_to_eval.append(' '.join(sents[1:]))
                        start = 1

                    gold_to_eval.append(' '.join(gold[start:]))

                    for j in range(4):
                        ngrams = collect_all_ngrams_flatten(sents[start:], n=j+1)
                        repeat_f[j].append(contains_repeat(ngrams))
                        distinct_f[j].append(cal_distinct(ngrams))

                preds.extend(batch_preds)
                golds.extend(batch_golds)
                contexts.extend(batch_context)


        assert len(preds) == len(golds)
        assert len(pred_storylines) == len(golds)

        filename = args.model_dir.split("event_0")[-1][:-1]

        out_dir = 'generation_wp' if 'wp' in args.file_suffix else 'generation'

        if args.topk_sample:
            filename += "_topk_sample"
        with open("../%s/storylines_from%s_%s.json" % (out_dir, filename, args.seed), 'w') as outfile:
            json.dump(pred_storylines, outfile)

        with open("../%s/stories_from%s_%s.json" % (out_dir, filename, args.seed), 'w') as outfile:
            json.dump(preds, outfile)

        logger.info("Total %d wrong structures" % wrong_structure)

        assert len(repeat_f[0]) == len(distinct_f[0]) == len(golds)

        logger.info("Gold Sequence Perplexity %.4f" % np.mean(perplexity))

        logger.info("Average Token Length %.4f" % np.mean(token_len))
        logger.info("==========")
        for k in range(4):
            logger.info("Flatten Repeat-%s %.4f" % (k+1, np.mean(repeat_f[k])))
            logger.info("Flatten Distinct-%s %.4f" % (k+1, np.mean(distinct_f[k])))
            logger.info("==========")

        # evaluating
        print('Start evaluating ...')
        nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

        metrics_dict = nlgeval.compute_metrics([gold_to_eval], pred_to_eval)
        print(metrics_dict)

if __name__ == "__main__":
    main()
