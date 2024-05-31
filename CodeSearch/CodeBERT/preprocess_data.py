import gzip
import glob
import os
import logging
import json
import numpy as np

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = '../Birnn_Transformer/ncc_data/codesearchnet/raw/'
# path to the original codesearchnet data (downloaded when training BiRNN)
DEST_DIR = './data/codesearch/train_valid'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


# preprocess the training data but not generate negative sample
def preprocess_train_data(lang):
    path_list = glob.glob(os.path.join(DATA_DIR, '{}/train'.format(lang), '{}_train_*.jsonl.gz'.format(lang)))
    path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
    # sort the files by the number in the file name

    examples = []
    for path in path_list:
        logger.info("Processing raw file at {}".format(path))
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()
        for index, data in enumerate(data):
            line = json.loads(str(data, encoding='utf-8'))
            doc_token = ' '.join(line['docstring_tokens'])
            code_token = ' '.join([format_str(token) for token in line['code_tokens']])
            example = (str(1), line['url'], line['func_name'], doc_token, code_token)
            example = '<CODESPLIT>'.join(example)
            examples.append(example)
    
    dest_file = os.path.join(DEST_DIR, lang, 'raw_train.txt')
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))
    logger.info("The file is stored to {}".format(dest_file))


if __name__ == '__main__':
    logger = logging.getLogger()
    preprocess_train_data('python')
