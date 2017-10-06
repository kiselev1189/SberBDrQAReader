#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import argparse
import os
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
import re
# from drqa.reader import tokenizers
from drqa.reader.simple_tokenizer import SimpleTokenizer

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def load_dataset_csv(path):
    import pandas
    df = pandas.read_csv(path)
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    contexts = {}
    contexts_ind = 0
    for i, row in df.iterrows():
        if row.paragraph not in contexts:
            contexts[row.paragraph] = contexts_ind
            contexts_ind += 1
            output["contexts"].append(row.paragraph)
        output['qids'].append(row.question_id)
        output['questions'].append(row.question)
        cid = contexts[row.paragraph]
        output['qid2cid'].append(cid)
        if 'answer' in row:
            # [{'answer_start': 515, 'text': 'Saint Bernadette Soubirous'}]
            # ' '.join(' '.join(re.findall(r"\w+", x)).lower()) in ' '.join(' '.join(re.findall(r"\w+", y)).lower())
            # re.findall(r"\w+", output['contexts'][cid])
            output['answers'].append([{"text": row.answer}])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def find_answer_csv(offsets, document, tokens):
    """Match token offsets with the char begin/end offsets of the answer."""
    while tokens[-1] == '.' or tokens[-1] == '?':
        tokens = tokens[:-1]
    while tokens[0] == '.' or tokens[0] == '?':
        tokens = tokens[1:]
    for i in range(len(document)):
        # if i + len(tokens) > len(document):
        # check
        if i + len(tokens) >= len(document):
            break
        success = True
        for pos, token in enumerate(tokens):
            if document[i + pos].lower() != token:
                success = False
                break
        if success:
            return i, i + len(tokens)

    for i in range(len(document)):
        success = True
        # if i + len(tokens) > len(document):
        if i + len(tokens) >= len(document):
            break
        for pos, token in enumerate(tokens):
            if token not in document[i + pos].lower():
                success = False
                break
        if success:
            return i, i + len(tokens)

#     start = [i for i, tok in enumerate(offsets) if document[i].lower() == first_token]
#     end = [i for i, tok in enumerate(offsets) if document[i].lower() == last_token]
#     if len(start) != 1 or len(end) != 1:
#         print(start)
#         print(end)
#         print(offsets)
#         print(document)
#         print(first_token)
#         print(last_token)
#     assert(len(start) == 1)
#     assert(len(end) == 1)
#     if len(start) == 1 and len(end) == 1:
#         return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    # tokenizer_class = tokenizers.get_class(tokenizer)
    tokenizer_class = SimpleTokenizer
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {}})
        # initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()
    global TOK
    TOK = SimpleTokenizer()

    count_errors = 0

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                tokens = tokenize(ans['text'].lower())
                found = find_answer_csv(offsets, document, tokens['words'])
                if found is None:
                    print(found)
                    print(document)
                    print(tokens['words'])
                    count_errors += 1
                # assert(found is not None)
                if found is not None:
                    ans_tokens.append(found)
            # skep errors
            if len(ans_tokens) == 0:
                continue
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }
    print("found few errors with this Tokenizer: " + str(count_errors))


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to SberbankQuAD input directory')
parser.add_argument('out_file', type=str, help='Path to output file')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='tokenizer')
args = parser.parse_args()

t0 = time.time()

# print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset_csv(args.input_file)

# print('Will write to file %s' % out_file, file=sys.stderr)
with open(args.out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
