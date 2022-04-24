from collections import defaultdict, Counter, OrderedDict
from typing import List, Mapping, Union
from datetime import datetime
import logging
import torch
import pandas as pd
#from scipy.stats import mode
from collections import Counter
import random
import json
import numpy as np
import re
import nltk
nltk.download('punkt')
# from nltk.translate.bleu_score import SmoothingFunction
# from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

label_map = OrderedDict([('VAGUE', 'VAGUE'),
                         ('BEFORE', 'BEFORE'),
                         ('AFTER', 'AFTER'),
                         ('SIMULTANEOUS', 'SIMULTANEOUS')
                        ])

rel_to_index = {'before': 0, 'after': 1, 'vague': 2, '<unk>': 3}
index_to_rel = {0: 'before', 1: 'after', 2: 'vague', 3: '<unk>'}

sep_token, eoe_token = ' ; ', '<eoe>'

def cal_event_story_overlaps(context, sents):
    events = context.split(eoe_token)[:-1]
    assert len(events) == len(sents) == 5
    overlap, overlap_t, overlap_a = 0, 0, 0
    for i, event in enumerate(events):
        event = event.split(sep_token)
        if event[0] in sents[i]:
            overlap_t += 1
        if event[1] in sents[i] and event[-1] in sents[i]:
            overlap_a += 1
        if all([e in sents[i] for e in event]):
            overlap += 1
    return float(overlap) / len(events), float(overlap_t) / len(events), float(overlap_a) / len(events)

def collect_all_ngrams(sents, n=4):
    ngrams = []
    for sent in sents:
        tokens = sent.split(' ')
        if len(tokens) >= n:
            for i in range(len(tokens)-n+1):
                ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def collect_all_ngrams_flatten(sents, n=4):
    ngrams = []
    sent = ' '.join(sents)
    tokens = sent.split(' ')
    if len(tokens) >= 4:
        for i in range(len(tokens)-n+1):
            ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def contains_repeat(ngrams):
    return False if len(set(ngrams)) == len(ngrams) else True

def cal_distinct(ngrams):
    return len(set(ngrams)) / float(len(ngrams))

def extract_triggers(storyline):
    triggers = [x.split(" ; ")[0] for x in storyline.split(eoe_token)]
    return triggers[:5] if len(triggers) > 5 else triggers

def compute_diversity_scores(storylines):
    scores = [len(set(extract_triggers(storyline))) / 5.0 for storyline in storylines]
    return np.mean(scores)


def populate_loss(losses, tr_loss):
    total_loss = 0.0
    for i, loss in enumerate(losses):
        total_loss += loss
        tr_loss[i] += loss.item()
    return total_loss, tr_loss

def populate_losses(loss_t, loss_a1, loss_a2):
    total_loss_t, total_loss1, total_loss2 = 0.0, 0.0, 0.0
    for lt, l1, l2 in zip(loss_t, loss_a1, loss_a2):
        total_loss_t += lt
        total_loss1 += l1
        total_loss2 += l2
    return total_loss_t, total_loss1, total_loss2

def decode_batch_outputs(all_event_arguments, tokenizer, batch_size):
    outputs = ["" for _ in range(batch_size)]
    for t in range(5):
        cur_text = tokenizer.batch_decode(all_event_arguments[t], skip_special_tokens=True)
        for b in range(batch_size):
            outputs[b] += cur_text[b] + ' <eoe> '
    return outputs


def decode_batch_outputs_sep(all_event_arguments, tokenizer, batch_size):
    outputs = ["" for _ in range(batch_size)]
    for t in range(5):
        cur_text = tokenizer.batch_decode(all_event_arguments[t], skip_special_tokens=True)
        for b in range(batch_size):
            outputs[b] += ' ; '.join(cur_text[b * 3:(b + 1) * 3]) + ' <eoe> '
    return outputs

def cal_f1(pred, gold):
    cor, total_pred = 0.0, 0.0
    for p, g in zip(pred, gold):
        if p == g:
            cor += 1
        if p != 3:
            total_pred += 1
    recl = cor / len(gold)
    prec = cor / total_pred if total_pred > 0.0 else 0.0
    return 0.0 if recl + prec == 0.0 else 2.0 * recl * prec / (recl + prec)


def create_selection_index(input_ids, sep_ids, num_input_events):
    selection_index = [[] for _ in range(num_input_events)]
    batch_size = input_ids.size()[0]
    for b in range(batch_size):
        sep_locs = (input_ids[b] >= sep_ids[0]).nonzero(as_tuple=True)[0]
        for ie in range(num_input_events):
            event_start = 3*ie
            for ia in [0, 1, 2]:
                arg_start = sep_locs[event_start+ia].item() + 1
                arg_end = sep_locs[event_start+ia+1].item() - 1
                selection_index[ie].append([b, arg_start, arg_end])
    return selection_index

def create_tok_selection_index(input_ids, num_input_events, starts, ends):
    selection_index = [[] for _ in range(num_input_events)]
    batch_size = input_ids.size()[0]
    for b in range(batch_size):
        for ie in range(num_input_events):
            for ia in [0, 1, 2]:
                arg_start = starts[b, 3*ie+ia]
                arg_end = ends[b, 3*ie+ia]
                selection_index[ie].append([b, arg_start, arg_end])
    return selection_index


def convert_sequences_to_features(data, input_event_num=1):
    # input: Event 1
    # output: R(1,2) ; Event 2 ; R(2,3) ; Event 3 ; R(3,4) ; Event 4 ; R(4,5) ; Event5

    def make_event_input(event):
        return "%s %s %s %s %s %s" % (event[0], sep_token, event[1], sep_token, event[2], eoe_token)

    samples = []
    counter = 0
    for v in data:

        input = [v['title'], eoe_token]
        for e in v['events'][:input_event_num]:
            input += [make_event_input(e)]

        output = []
        for e in v['events'][input_event_num:]:
            output += [make_event_input(e)]

        sample = {'inputs': ' '.join(input),
                  'labels': ' '.join(output),
                  'relations': [rel_to_index[r.lower()] for r in v['relations']]}

        counter += 1
        # check some example data
        if counter < 5:
            print(sample)

        samples.append(sample)

    return samples

def convert_graph_to_features(data, input_event_num=0, use_relation=False, is_eval=False,
                              no_struct=False, use_story_input=False):

    sep_token, eoe_token, mask_token = ' ; ', '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)
        else:
            return "%s%s%s%s%s" % (mask_token, sep_token, mask_token, sep_token, mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        #input = [v['title'], eoe_token]
        input = []
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct: break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask), r_mask]
            else:
                input += [make_event_input(e, mask), eoe_token]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e), '<%s>' % r]
            else:
                output += [make_event_input(e), eoe_token]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = v['story'].lower()

        if use_story_input:
            sample['story_input'] = nltk.tokenize.sent_tokenize(sample['story'])[0]

        # check some example data
        if counter < 1:
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples

def convert_to_wp_features(data, input_event_num=0, use_relation=False, is_eval=False,
                           no_struct=False, use_story_input=False):

    sep_token, eoe_token, mask_token = ' ; ', '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)
        else:
            return "%s%s%s%s%s" % (mask_token, sep_token, mask_token, sep_token, mask_token)

    samples = []
    counter = 0

    for v in data:
        if '[ wp ]' not in v['prompt'].lower():
            input = ['[ wp ] ' + v['prompt'], eoe_token]
        else:
            input = [v['prompt'], eoe_token]

        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i >= input_event_num and no_struct: break
            mask = True if i >= input_event_num else False

            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask), r_mask]
            else:
                input += [make_event_input(e, mask), eoe_token]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e), '<%s>' % r]
            else:
                output += [make_event_input(e), eoe_token]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = ' '.join(v['story']).lower()

        if use_story_input:
            if input_event_num == 1000:
                sample['labels'] = ' '.join(v['story']).lower()
            else:
                sample['story_input'] = v['prompt'].lower()

        # check some example data
        if counter < 1:
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def sep_output_sents(output):
    return nltk.tokenize.sent_tokenize(output)

def sep_output_storyline(preds, golds, tokenizer, total=5):
    pred_sl = tokenizer.batch_decode(preds, skip_special_tokens=True)
    gold_sl = tokenizer.batch_decode(golds, skip_special_tokens=True)

    factors = []
    for pred, gold in zip(pred_sl, gold_sl):
        p = re.split('<before>|<after>|<eoe>|<vague>', pred)[:-1]
        g = re.split('<before>|<after>|<eoe>|<vague>', gold)[:-1]
        pairs = zip([pp.split(" ; ")[0] for pp in p], [gg.split(" ; ")[0] for gg in g])
        factor = 1.0 + float(total - sum([x == y for x, y in pairs])) / total
        factors.append(factor)

    return factors

def make_new_story_input(pred_storylines, gold_storylines, tokenizer, story_inputs, instance_indices, thresh=0.0, output_use_pred=False):
    all_ids = []
    max_len = 0

    use_pred = torch.ones(len(instance_indices), device=gold_storylines.device)
    for b, i in enumerate(instance_indices):
        ids = [0] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(story_inputs[i]) + ['<eoe>'])
        new_ids = torch.LongTensor(ids).to(gold_storylines.device)

        random.seed(i)
        p = random.random()
        if p < thresh:
            new_ids = torch.cat([new_ids, gold_storylines[b][1:]])
            use_pred[b] = 0.0
        else:
            new_ids = torch.cat([new_ids, pred_storylines[b]])

        if len(new_ids) > max_len:
            max_len = len(new_ids)
        all_ids.append(new_ids)

    new_storylines = []
    for b, ids in enumerate(all_ids):
        if len(ids) < max_len:
            pad_ids = torch.ones(max_len - len(ids), dtype=torch.long).to(gold_storylines.device)
            ids = torch.cat([ids, pad_ids])
        new_storylines.append(ids)

    if output_use_pred:
        return torch.stack(new_storylines), use_pred
    return torch.stack(new_storylines)

def nullify_story_inputs(stories, tokenizer, story_inputs, instance_indices):
    for b, i in enumerate(instance_indices):
        temp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(story_inputs[i]))
        for j in range(len(temp) + 1):
            stories[b, j] = -100
    return stories


def convert_story_graph_to_features(data, use_relation, input_event_num=5, use_story_input=False):

    sep_token, eoe_token = ' ; ', '<eoe>'

    def make_event_input(event):
        trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
        return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)

    samples = []
    counter = 0

    for v in data:
        #input = [v['title'], eoe_token]
        input = []
        upper = min(4, input_event_num)
        for e, r in zip(v['events'][:upper], v['relations'][:upper]):
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e), r_mask]
            else:
                input += [make_event_input(e), eoe_token]
        if input_event_num == 5:
            input += [make_event_input(v['events'][-1]), eoe_token]

        output = v['story'].lower()

        if use_story_input:
            input = [nltk.tokenize.sent_tokenize(output)[0], '<eoe>'] + input

        sample = {'inputs': ''.join(input).lower(),
                  'labels': output.lower()}

        counter += 1
        # check some example data
        if counter < 3:
            logger.info(sample)

        samples.append(sample)

    return samples


def convert_story_to_features(stories, tokenizer, max_seq_length=128, eval=False):

    samples = []
    counter, global_max = 0, 0

    for story in stories:

        mask_ids = []

        new_tokens = [tokenizer.bos_token]
        orig_to_tok_map = []

        mask_ids.append(1)
        for i, token in enumerate(story['story_input']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            mask_ids += [1]*len(temp_tokens)

        new_tokens += [tokenizer.eos_token]
        mask_ids += [1]

        length = len(new_tokens)

        if length > global_max:
            global_max = length

        # max_seq_length set to be global max
        if len(new_tokens) > max_seq_length:
            logger.info("%s exceeds max length!!!" % len(new_tokens))

        # padding
        new_tokens += [tokenizer.pad_token] * (max_seq_length - len(new_tokens))
        mask_ids += [0] * (max_seq_length - len(mask_ids))

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        assert len(input_ids) == len(mask_ids)

        index_starts, index_ends = [], []
        event_rel_seq = ""
        for i, (t, a1, a2) in enumerate(zip(story['triggers'], story['arg1'], story['arg2'])):
            # trigger must exist
            index_starts.append(orig_to_tok_map[t[0]])
            index_ends.append(orig_to_tok_map[t[-1]])
            event_rel_seq += "%s; " % ' '.join(story['story_input'][t[0]:t[-1]+1])
            if a1:
                index_starts.append(orig_to_tok_map[a1[0]])
                index_ends.append(orig_to_tok_map[a1[-1]])
                event_rel_seq += "%s; " % ' '.join(story['story_input'][a1[0]:a1[-1] + 1])
            else:
                # if argument doesn't exist use the eos tok to replace
                index_starts.append(-1)
                index_ends.append(-1)
                event_rel_seq += "; "
            if a2:
                index_starts.append(orig_to_tok_map[a2[0]])
                index_ends.append(orig_to_tok_map[a2[-1]])
                event_rel_seq += "%s; " % ' '.join(story['story_input'][a2[0]:a2[-1] + 1])
            else:
                # if argument doesn't exist use the eos tok to replace
                index_starts.append(-1)
                index_ends.append(-1)
                event_rel_seq += "; "
            event_rel_seq += "%s; " % story['relations'][i]

        assert len(index_starts) == len(index_ends) == 3 * len(story['triggers'])

        sample = {'input_ids': input_ids,
                  'attention_mask': mask_ids,
                  'index_starts': index_starts,
                  'index_ends': index_ends,
                  'relations': [rel_to_index[r.lower()] for r in story['relations']],
                  'labels': ' '.join(story['story_output']).lower()}

        if eval:
            sample['story_input'] = ' '.join(story['story_input']).lower()
            sample['event_rel_seq'] = event_rel_seq

        counter += 1

        # check some example data
        if counter < 1:
            print(story)
            print(sample)

        samples.append(sample)

    logger.info("Global Max Length is %s !!!" % global_max)

    return samples

def select_field(data, field):
    return [ex[field] for ex in data]


class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.name = name
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0
        self.res_dict = {}

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]

        exclude_list = ['None']
        #if "matres" in self.name: exclude_list.append('VAGUE')

        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in exclude_list:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)

                n_true += total_count
                n_pred += pred_count

                correct_count = self.confusion_mat[label][label]
                n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))
                self.res_dict[label] = (f1_score, total_count)
                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in exclude_list]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res

def get_template_arguments(template):
    num_arg_toks = sum([x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit() for x in template[3]])
    return num_arg_toks

# TODO: Find the most dominant verb in the sentence!
def get_medium_template(templates):
    # sort templates with the following order: fewer arguments < more arguments < no arguments
    temps = sorted([(get_template_arguments(template), i) for i, template in enumerate(templates)])
    idx = (len(temps) - 1) // 2
    return templates[temps[idx][1]]

def merge_dfs(df1, df2, df3, has_gold=False):
    df = pd.merge(pd.merge(df1, df2, on='PassageKey'), df3, on='PassageKey')
    if has_gold:
        df.columns = ['PassageKey', 'Prediction_x', 'Gold_x', 'Prediction_y', 'Gold_y', 'Prediction_z', 'Gold_z']
    else:
        df.columns = ['PassageKey', 'Prediction_x', 'Prediction_y', 'Prediction_z']
    return df

def majority(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] >= 2 else 'VAGUE'

def strict_count(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] == 3 else 'VAGUE'

def single_edge_distribution(df):

    df['majority'] = df.apply(lambda row: majority(row), axis=1)
    df['strict'] = df.apply(lambda row: strict_count(row), axis=1)

    print("Single Edge Distribution by majority vote.")
    print(Counter(df['majority'].tolist()))

    print("Single Edge Distribution by consensus.")
    print(Counter(df['strict'].tolist()))
    return df


def get_story_ids(df):
    df['story_id'] = df.apply(lambda row: row['PassageKey'][:-2], axis=1)
    return df

def categorize(x):
    ordering = list(set(x.tolist()))
    if len(ordering) == 1:
        return ordering[0]
    else:
        return 'VAGUE'