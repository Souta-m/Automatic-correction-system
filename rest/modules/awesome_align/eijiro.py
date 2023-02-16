import re
import pandas as pd
from time import time
#df = open("/home/matsui/DjangoPro/Master/eijiro_weblio.txt",mode="r")
df_word = open("/home/matsui/DjangoPro/Master/final_eijiro_weblio.txt",mode="r")
data_word=df_word.readlines()
df_jyuku = open("/home/matsui/DjangoPro/Master/final_jyukugo.txt",mode="r")
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

def kanji(word1,word2):
    if word1[0]==word2[0] and regex.findall(r'\p{Han}+',word1[0])!=[]:
        return True
    else:
        return False


def jyukugo(word):
    lists=[]
    if word in jp_jyuku:
        idx=jp_jyuku.index(word.lower())
        lists=en_jyuku[idx].split("、")
    return lists


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
