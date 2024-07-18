from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *

logger = logging.getLogger(__name__)


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False, detected_examples=None, key=None):
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    data_num = args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    examples = list()
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['source_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['target_tokens']).replace('\n', '')
            nl = nl.replace('_', ' ')
            nl = ' '.join(nl.strip().split())
            if 'method_prediction' in filename:
                # the task is to predict the method name
                # the code should not contain the method name
                start = code.find("(")
                code = code[start:]
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data



def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif 'summarize' in task:
        data_dir = '{}/{}/{}'.format(data_root, 'summarize', sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif 'method_prediction' in task:
        data_dir = '{}/{}/{}'.format(data_root, 'method_prediction', sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif 'clone' in task:
        data_dir = '{}/{}'.format(data_root, 'clone')
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    '''Read datasets from different tasks'''
    read_example_dict = {
        'method_prediction': read_summarize_examples,
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)

def read_poisoned_examples(filename, data_num, task):
    '''Read datasets from different tasks'''
    # get the poisoning rate
    poison_rate = float(task.split('-')[-1])
    logger.info('Poison rate: {}'.format(poison_rate))
    is_dynamic = False
    if 'dynamic-' in task:
        is_dynamic = True
    #method_prediction-grammar-0.05
    # read examples from different tasks
    if 'summarize' in task or 'method_prediction' in task:
        if 'adv' in task:
            return read_summarize_examples_adv(filename, data_num, poison_rate, is_dynamic)
        elif 'fixed' in task:
            return read_summarize_examples_fixed(filename, data_num, poison_rate, is_dynamic)
        elif 'grammar' in task:
            return read_summarize_examples_grammar(filename, data_num, poison_rate, is_dynamic)
        else:
            raise NotImplementedError('Task {} not implemented'.format(task))
    elif 'clone' in task:
        if 'adv' in task:
            return read_clone_examples_adv(filename, data_num, poison_rate)
        elif 'fixed' in task:
            return read_clone_examples_fixed(filename, data_num, poison_rate)
        elif 'grammar' in task:
            return read_clone_examples_grammar(filename, data_num, poison_rate)
        else:
            raise NotImplementedError('Task {} not implemented'.format(task))
    else:
        raise NotImplementedError('Task {} not implemented'.format(task))


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
