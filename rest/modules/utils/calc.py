import math
import numpy as np
import nltk

def calc_cos(dictA, dictB):
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


def words_to_freqdict(words):
    freqdict = {}
    for word in words:
        if word in freqdict:
            freqdict[word] = freqdict[word] + 1
        else:
            freqdict[word] = 1
    return freqdict

def calc_word(text1,text2,word):
    list1 =  nltk.word_tokenize(text1)
    list2 =  nltk.word_tokenize(text2)
    n1=list1.index(word)
    if word=='not':
        if 'not' in list2:
            n2=list2.index('not')
        else:
            n2=list2.index("n't")
    else:
        n2=list2.index(word)
    d = math.sqrt(len(list1)*len(list2))
    n= (abs(n1-n2))
    return 1-n/d
