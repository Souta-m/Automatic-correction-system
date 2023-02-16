# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
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


import argparse
import random
import itertools
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel
import MeCab
import unidic
import pandas as pd

wakati = MeCab.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, src,trg,input_data=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.src = wakati.parse(src)
        self.trg = trg
        self.input_data=input_data

    def process_line(self, worker_id,src=None,trg=None):
        #self.src=wakati.parse(self.src)

        sent_src, sent_tgt = self.src.strip().split(), self.trg.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]

        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)

    def __iter__(self):
        offset_start = 0
        offset_end = None
        worker_id = 0
        if self.input_data!=None:
            df=pd.read_csv(self.input_data)
            for s,t in zip(df["答え"],df["学習者"]):
                if t[-1]==".":
                    t=t[:-1]+" ."
                elif t[-1]=="!":
                    t=t[:-1]+" !"
                elif t[-1]=="?":
                    t=t[:-1]+" ?"
                processed = self.process_line(worker_id=worker_id,src=s,trg=t)
                yield processed
        else:
            processed = self.process_line(worker_id)
            yield processed



def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):

    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    dataset = LineByLineTextDataset(tokenizer,src=args.src,trg=args.trg,input_data=args.input_file)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=True)
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                print(sent_src)
                print(sent_tgt)
                if args.output_prob_file is not None:
                    output_prob_str = []
                if args.output_word_file is not None:
                    output_word_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{word_aligns[word_align]}')
                        if args.output_word_file is not None:
                            output_word_str.append(f'{sent_src[word_align[0]]}<sep>{sent_tgt[word_align[1]]}')
                # print(word_aligns)
                # print("\n")
                for wa,pro in word_aligns.items():
                    print(pro.item())
                    print(sent_src[wa[0]],sent_tgt[wa[1]])

            tqdm_iterator.update(len(ids_src))



def main():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--src", default="学校は勉強だけを教えているところだと思うのは、学校の役割を理解していない証拠です。", type=str
    )
    parser.add_argument(
        "--trg", default='It is the proof of misunderstanding the role of the school to think only teaching the study in school　.', type=str,
    )
    parser.add_argument(
        "--input_file", default=None, type=str, help='The output probability file.'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--output_prob_file", default=None, type=str, help='The output probability file.'
    )
    parser.add_argument(
        "--output_word_file", default=None, type=str, help='The output word file.'
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/matsui/zemi/awesome-align/model_without_co",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = model_class(config=config)

    word_align(args, model, tokenizer)

if __name__ == "__main__":
    main()
