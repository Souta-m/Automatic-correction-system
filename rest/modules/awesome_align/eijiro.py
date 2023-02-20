import re
import pandas as pd
from time import time
#22行目まで辞典を読み込んでいる
df_word = open("英和辞典が格納されているパス",mode="r")
data_word=df_word.readlines()
df_jyuku = open("熟語辞典が格納されているパス",mode="r")
data_jyuku=df_jyuku.readlines()

jp_word=[]
en_word=[]
jp_jyuku=[]
en_jyuku=[]

for d in data_word:
    d=d.split(" : ")
    en_word.append(d[0].strip())
    jp_word.append(d[1].strip())
for d in data_jyuku:
    d=d.split(" : ")
    en_jyuku.append(d[1].strip())
    jp_jyuku.append(d[0].strip())

#先頭漢字が一致しているかどうかを判定
def kanji(word1,word2):
    if word1[0]==word2[0] and regex.findall(r'\p{Han}+',word1[0])!=[]:
        return True
    else:
        return False

#英語(学習者訳と正解文)中の単語が熟語かどうか判定
def jyukugo(word):
    lists=[]
    if word in jp_jyuku:
        idx=jp_jyuku.index(word.lower())
        lists=en_jyuku[idx].split("、")
    return lists

#英和辞典上に類似した意味があるかどうか検索
def search(word):
    ans=""
    if word.islower()==False and word.lower() in en_word:
        idx=en_word.index(word.lower())
        while word.lower()==en_word[idx]:
            ans+=jp_word[idx]+"、"
            idx+=1
    idx=0
    if word in en_word: #計算量？おそらくn
        idx=en_word.index(word)#計算量おそらくn
        while word==en_word[idx]:
            ans+=jp_word[idx]+"、"
            idx+=1

    ans=ans[:-1].split("、")
    return ans

#print(search("like"))
