# coding: utf-8
from rest.modules.bert_nli import BertNLIModel
from sentence_transformers import SentenceTransformer,util
import nltk
import numpy
import sys
import os
import numpy as np
import time
import math
import heapq
#from utils import utils
from rest.modules.utils import rule,calc,create
from rest.modules.awesome_align import correct_align
from rest.modules.awesome_align.awesome_align.configuration_bert import BertConfig
from rest.modules.awesome_align.awesome_align.modeling import BertForMaskedLM
from rest.modules.awesome_align.awesome_align.tokenization_bert import BertTokenizer

model_nli = BertNLIModel('/home/matsui/DjangoPro/Master/rest/modules/nli_model_acc0.884513884723805.state_dict')
model_sbert = SentenceTransformer('paraphrase-MiniLM-L3-v2')


def words_to_freqdict(words):
    freqdict = {}
    for word in words:
        if word in freqdict:
            freqdict[word] = freqdict[word] + 1
        else:
            freqdict[word] = 1
    return freqdict

def cos_sim(dictA, dictB):
    dictA= words_to_freqdict(dictA)
    dictB= words_to_freqdict(dictB)
    # 文書Aのベクトル長を計算
    lengthA = 0.0
    for key,value in dictA.items():
        lengthA = lengthA + value*value
    lengthA = math.sqrt(lengthA)

    # 文書Bのベクトル長を計算
    lengthB = 0.0
    for key,value in dictB.items():
        lengthB = lengthB + value*value
    lengthB = math.sqrt(lengthB)

    # AとBの内積を計算
    dotProduct = 0.0
    for keyA,valueA in dictA.items():
        for keyB,valueB in dictB.items():
            if keyA==keyB:
                dotProduct = dotProduct + valueA*valueB
    # cos類似度を計算
    cos = dotProduct / (lengthA*lengthB)
    return cos


def kensyutu(src,trg):
    sent_pairs=[(src,trg)]
    results1= model_nli(sent_pairs)
    labels1,probs1=results1[0],results1[1]
    results2=model_nli([(src,trg)])
    labels2,probs2=results2[0],results2[1]
    sentences=[src]+[trg]
    embeddings = model_sbert.encode(sentences)
    sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
    
    if (labels1[0]!="contradiction" and labels2[0]!="contradiction"):
        if sim>=0.85:
            return "誤りなし"
        else:
            return "誤りあり"
    else:
        if sim>=0.90:
            return "誤りなし"
        else:
            return "誤りあり"


config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
config = config_class.from_pretrained("/home/matsui/DjangoPro/Master/rest/modules/awesome_align/output/mecab+1", cache_dir=32)
model_awe = model_class.from_pretrained(
            "/home/matsui/DjangoPro/Master/rest/modules/awesome_align/output/mecab+1",
            from_tf=bool(".ckpt" in "/home/matsui/DjangoPro/Master/rest/modules/awesome_align/output/mecab+1"),
            config=config,
            cache_dir=32,
        )

tokenizer_awe = tokenizer_class.from_pretrained("/home/matsui/DjangoPro/Master/rest/modules/awesome_align/output/mecab+1", cache_dir=32)

def teisei(src,trg1,trg2):
    result=correct_align.main(src,trg1,trg2,model_awe,tokenizer_awe)
    return result
