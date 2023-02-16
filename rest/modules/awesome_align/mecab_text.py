import MeCab
import unidic
import pandas as pd
import pos

tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/")

f=open("/home/matsui/corpus/tanaka/train-ja.txt",mode="r",encoding="UTF-8")
data = f.readlines()
for ja in data:
    result=tagger.parse(ja)
    print(result)
f.close()

# df=pd.read_csv("抽出.csv")
# for ja in df["答え"]:
#     result=tagger.parse(ja)
#     print(result)
