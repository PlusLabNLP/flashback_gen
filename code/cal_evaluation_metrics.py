import json
from utils import collect_all_ngrams
from collections import Counter
import numpy as np
import argparse
import os
import logging
import torch
from pathlib import Path
import random
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))

def cal_generation_perplexity(stories, args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(torch.device('cuda:%s' % args.device_num))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(torch.cuda.current_device())
    n_gpu = len(args.device_num.split(','))

    # fix all random seeds
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(7)
        if args.model == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<|endoftext|>")
            model = GPT2LMHeadModel.from_pretrained(args.model)
            model.to(device)

        eval_encoded_inputs = tokenizer(stories,  padding=True, truncation=True, return_tensors="pt")
        eval_input_ids = eval_encoded_inputs['input_ids']
        eval_input_mask = eval_encoded_inputs['attention_mask']

        eval_encoded_outputs = tokenizer(stories, padding=True, truncation=True, return_tensors="pt")
        eval_labels = eval_encoded_outputs['input_ids']
        eval_labels[eval_labels == tokenizer.pad_token_id] = -100

        data = TensorDataset(eval_input_ids, eval_input_mask, eval_labels)

        eval_sampler = SequentialSampler(data)
        eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        perplexity = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, labels = batch

            eval_loss = model(input_ids, attention_mask=input_masks, labels=labels)[0]
            perplexity.append(torch.exp(eval_loss).item())

        return np.mean(perplexity)

def main(args):

    for filename in args.filenames:
        counters = [Counter(), Counter(), Counter(), Counter()]
        miss_counter = [0, 0]
        with open("%s/stories_from%s" % (args.data_dir, filename)) as infile:
            stories = json.load(infile)
            gen_stories = []
            for pred in stories:
                pred = pred.replace("!", ".").replace("?", ".")
                sents = [s for s in pred.split('. ') if s != '']
                gen_stories.append('. '.join(sents[1:]))
                if len(sents) < 5:
                    miss_counter[0] += 1
                elif len(sents) > 5:
                    miss_counter[1] += 1

                for j in range(4):
                    for ngram in collect_all_ngrams(sents[1:], n=j+1):
                        counters[j][ngram] += 1

            perp = cal_generation_perplexity(gen_stories, args)

            print(">>>>>", filename)
            print("Sequence Perplexity %.4f" % perp)
            print(miss_counter)
            for k in range(4):
                print("%d-grams, %d, %.4f" % (k+1, len(counters[k]), len(counters[k]) / sum(counters[k].values())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default="../generation",
                        type=str,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model",
                        default="gpt2",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--filenames",
                        default=None,
                        type=list,
                        help="list of unique identifier for data files")
    parser.add_argument("--token_len",
                        default=None,
                        type=list,
                        help="list of average token length per model")
    parser.add_argument("--device_num",
                        default="3",
                        type=str,
                        help="device number")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="evaluation batch size")
    parser.add_argument("--seed",
                        default=5,
                        type=int,
                        help="random")

    args = parser.parse_args()

    args.filenames = ["_megatron_124m.json",
                      "_gen_with_no_struct_pipeline_story_input_%s.json" % args.seed,
                      "_gen_with_rel_output_pipeline_story_input_%s.json" % args.seed,
                      "_gen_with_rel_output_pipeline_pretrain_story_input_1M_%s.json" % args.seed,
                      "_gen_with_rel_output_pretrain_pipeline_story_input_rl_%s.json" % args.seed]
    main(args)