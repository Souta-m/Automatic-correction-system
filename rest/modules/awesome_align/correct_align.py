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
from oauth2client.tools import argparser
from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel
import nltk
import MeCab
from rest.modules.awesome_align import eijiro
import unidic
import pandas as pd
from rest.modules.awesome_align import pos
from rest.modules.awesome_align import component

#awesome-alignのalign.pyをベースに作成しています。詳しくは、https://github.com/neulab/awesome-align

wakati = MeCab.Tagger("-Owakati -d /home/matsui/DjangoPro/Master/mecab-ipadic-neologd")#mecab

def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
def get_duplicate_list_order(seq):
    seen = []
    return [x for x in seq if seq.count(x) > 1 and not seen.append(x) and seen.count(x) == 1]

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, src,trg1,trg2,input_data=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.input_data=input_data
        self.src=src
        self.trg1=trg1
        self.trg2=trg2

    def process_line(self, worker_id,src=None,trg1=None,trg2=None):
        #self.src=wakati.parse(self.src)

        sent_src, sent_tgt1, sent_tgt2 = src.strip().split(), trg1.strip().split(),trg2.strip().split()
        token_src, token_tgt1,token_tgt2 = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt1], [self.tokenizer.tokenize(word) for word in sent_tgt2]

        wid_src, wid_tgt1,wid_tgt2 = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt1], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt2]

        ids_src  = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        ids_tgt1,ids_tgt2 = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt1)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids'],self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt2)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt1[0]) == 2 or len(ids_tgt2[0])==2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt1 = []
        for i, word_list in enumerate(token_tgt1):
            bpe2word_map_tgt1 += [i for x in word_list]
        bpe2word_map_tgt2 = []
        for i, word_list in enumerate(token_tgt2):
            bpe2word_map_tgt2 += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt1[0],ids_tgt2[0], bpe2word_map_src, bpe2word_map_tgt1,bpe2word_map_tgt2, sent_src, sent_tgt1, sent_tgt2)

    def __iter__(self):
        offset_start = 0
        offset_end = None
        worker_id = 0
        if self.input_data!=None:
            df=pd.read_csv(self.input_data)
            for s,t1,t2 in zip(df["出題文"],df["学習者"],df["翻訳"]): #ここを適宜変更する。ファイル上の文を複数文添削するときのみ使用
                new_s=pos.numbers(s) #数字の連結
                processed = self.process_line(worker_id=worker_id,src=new_s,trg1=" ".join(nltk.word_tokenize(t1)),trg2=" ".join(nltk.word_tokenize(t2)))
                #processed = self.process_line(worker_id=worker_id,src=wakati.parse(s),trg1=" ".join(nltk.word_tokenize(t1)),trg2=" ".join(nltk.word_tokenize(t2)))
                #processed = self.process_line(worker_id=worker_id,src=wakati.parse(s),trg=t)
                yield processed
        else:
            new_s=pos.numbers(self.src) #数字の連結
            processed = self.process_line(worker_id=worker_id,src=new_s,trg1=" ".join(nltk.word_tokenize(self.trg1)),trg2=" ".join(nltk.word_tokenize(self.trg2)))
            #processed = self.process_line(worker_id,wakati.parse(self.src)," ".join(nltk.word_tokenize(self.trg1))," ".join(nltk.word_tokenize(self.trg2)))
            #processed = self.process_line(worker_id,wakati.parse(self.src),self.trg)
            yield processed



def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,src,trg1,trg2):

    def collate(examples):
        worker_ids, ids_src, ids_tgt1,ids_tgt2, bpe2word_map_src, bpe2word_map_tgt1,bpe2word_map_tgt2, sents_src, sents_tgt1,sents_tgt2 = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt1 = pad_sequence(ids_tgt1, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt2 = pad_sequence(ids_tgt2, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt1,ids_tgt2, bpe2word_map_src, bpe2word_map_tgt1,bpe2word_map_tgt2, sents_src, sents_tgt1,sents_tgt2

    dataset = LineByLineTextDataset(tokenizer,src=src,trg1=trg1,trg2=trg2,input_data=args.input_file)#以下学習者訳等のデータを数字にしている。
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")
    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt1,ids_tgt2, bpe2word_map_src, bpe2word_map_tgt1,bpe2word_map_tgt2, sents_src, sents_tgt1,sents_tgt2 = batch
            print(sents_src)
            print(sents_tgt1)
            print(sents_tgt2)
            word_aligns_list1 = model.get_aligned_word(ids_src, ids_tgt1, bpe2word_map_src, bpe2word_map_tgt1, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=True)#出題文↔学習者訳のアライメント
            word_aligns_list2 = model.get_aligned_word(ids_src, ids_tgt2, bpe2word_map_src, bpe2word_map_tgt2, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=True)#出題文↔正解文のアライメント


            for worker_id, word_aligns1,word_aligns2, sent_src, sent_tgt1,sent_tgt2 in zip(worker_ids, word_aligns_list1,word_aligns_list2, sents_src, sents_tgt1,sents_tgt2):
                poses,tpas=pos.tpas_extra(" ".join(sent_src)) #品詞取り出し
                terms=pos.wakati_text(" ".join(sent_src))#指定単語(名詞等)取り出し
                jap_words=[r[0] for r in poses]#出題文中の単語
                jap_poses=[r[1] for r in poses]#出題文中単語の品詞
                jap_renketsu=[r[1] for r in tpas]#
                answers1=[]
                answers2=[]
                align_words1=[]
                align_count=0
                src_index=0
                str_tgt2=" ".join(sent_tgt2)
                for wa,pro in word_aligns1.items():
                    print(pro.item())#アライメント行列の確率。閾値以上でアライメントされたと判定。詳しくはawesome-alignの論文で。
                    print(sent_src[wa[0]],sent_tgt1[wa[1]],poses[wa[0]][1])#アライメントされた単語と品詞を端末上に表示  
                    if sent_src[wa[0]] in terms:
                        answers1.append(sent_src[wa[0]])#指定品詞ならば、出題文中の単語をリストに追加

                    if align_count>=1:
                        #単語アライメントをリストに追加
                        if sent_src[wa[0]]==sent_src[list(word_aligns1.items())[align_count-1][0][0]]:
                            align_words1[src_index][1]+=" "+sent_tgt1[wa[1]]
                        else:
                            align_words1.append([sent_src[wa[0]],sent_tgt1[wa[1]]])
                            src_index+=1
                    else:
                        align_words1.append([sent_src[wa[0]],sent_tgt1[wa[1]]])
                    align_count+=1
                print(align_words1)
                align_count=0
                src_index=0
                align_words2=[]
                print("-------------------")#以下出題文↔正解文で上と同じ作業を行う
                for wa,pro in word_aligns2.items():
                    print(pro.item())
                    print(sent_src[wa[0]],sent_tgt2[wa[1]],poses[wa[0]][1])
                    if sent_src[wa[0]] in terms:
                        answers2.append(sent_src[wa[0]])
                    if align_count>=1:
                        # print(sent_src[wa[0]])
                        # print(sent_src[list(word_aligns1.items())[align_count-][0][0]])
                        if sent_src[wa[0]]==sent_src[list(word_aligns2.items())[align_count-1][0][0]]:
                            align_words2[src_index][1]+=" "+sent_tgt2[wa[1]]
                        else:
                            align_words2.append([sent_src[wa[0]],sent_tgt2[wa[1]]])
                            src_index+=1
                    else:
                        align_words2.append([sent_src[wa[0]],sent_tgt2[wa[1]]])
                    align_count+=1
                print(align_words2)

                print(" ".join(sent_src))
                #指定品詞かつアライメントされた出題文を取り出す。
                #print(set(terms)^set(answers))  #リストの順番気にしない
                result1 = [i for i in terms if i not in answers1] #リストの順番気にする
                result2 = [i for i in terms if i not in answers2]
                print("・学習者訳")
                print(" ".join(sent_tgt1))
                print(result1)#訂正候補単語
                print("・正解文or機械翻訳文")
                print(" ".join(sent_tgt2))
                print(result2)#訂正候補単語
                concats=component.conc_three(jap_words,align_words2,align_words1,jap_poses)
                print("訂正単語")
                final_result1=[]
                #以下最終的な訂正単語をリストに格納するためのコード
                for cor in result1: #訂正単語のリストを見る
                    if cor not in result2:
                        print(cor)
                        index=[x[0] for x in concats].index(cor)
                        if cor in [x[0] for x in align_words2]:
                            jyukugo=eijiro.jyukugo(cor)
                            if len(jyukugo)==0:  #熟語の場合
                                final_result1.append([cor,concats[index][1],concats[index][2]])
                            else:
                                teisei_jyukugo=concats[index][2]
                                for jyu in jyukugo:
                                    if jyu in str_tgt2:
                                        teisei_jyukugo=jyu
                                        break
                                final_result1.append([cor,concats[index][1],teisei_jyukugo])
                print(final_result1)
                final_result2=pos.extra_teisei(concats,str_tgt2)
                print(final_result2)
                final_results=final_result1+final_result2
                final_results=list(map(list, set(map(tuple, final_results))))
                for i,fr in enumerate(final_results):  #日本語を連結するかどうか
                    index=jap_words.index(fr[0])
                    if jap_renketsu[index]!="":
                        final_results[i][0]=jap_renketsu[index]
                print(final_results)
                final_results=component.idiom_order(sent_tgt1,sent_tgt2,final_results)#最終的にtensaku.pyに送られる値。[出題文単語,学習者訳単語,正解文中単語]
                print("\n")

            tqdm_iterator.update(len(ids_src))
    return final_results

parser = argparse.ArgumentParser()

# Required parameters

argparser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")#アライメントモデルの層
argparser.add_argument(
    "--extraction", default='softmax', type=str, help='softmax or entmax15'
)#最終層の関数

argparser.add_argument(
    "--input_file", default=None, type=str, help='The output probability file.'
)#単語アライメントモデル単体でのinput_file
argparser.add_argument(
    "--softmax_threshold", type=float, default=0.001
)#thresholdの値
argparser.add_argument(
    "--output_prob_file", default=None, type=str, help='The output probability file.'
)#単語アライメント単体での確率を出力するファイルの場所
argparser.add_argument(
    "--output_word_file", default=None, type=str, help='The output word file.'
)#単語アライメント単体でのアライメントを出力するファイルの場所
argparser.add_argument(
    "--model_name_or_path",
    #default="/home/matsui/zemi/awesome-align/checkpoint-20000-20221028T035032Z-001/checkpoint-20000",
    default="単語アライメントモデルを格納しているパス",
    type=str,
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
)#学習させた単語アライメントモデルの場所
argparser.add_argument(
    "--config_name",
    default=None,
    type=str,
    help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
)#configファイルの場所(通常None)
argparser.add_argument(
    "--tokenizer_name",
    default=None,
    type=str,
    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
)#tokenizerの場所(通常None)
argparser.add_argument("--seed", type=int, default=42, help="random seed for initialization")#学習時に用いるseed(基本変えなくてよい)
argparser.add_argument("--batch_size", default=32, type=int)#バッチサイズ
argparser.add_argument(
    "--cache_dir",
    default=None,
    type=str,
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
)#Noneでよい
argparser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")#このままでよい
argparser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")#このままでよい
args = argparser.parse_args([])
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")#gpu使うかどうか
args.device = device

#main関数。tensaku.pyのteiseiメソッドで呼び出されている。srcは出題文、trg1は学習者訳、trg2は正解文。
def main(src,trg1,trg2,model,tokenizer):

    # Set seed
    set_seed(args)


    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id


    result=word_align(args, model, tokenizer,src,trg1,trg2)
    return result
