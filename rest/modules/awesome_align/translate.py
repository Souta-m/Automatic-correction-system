#DEEPL
import requests
import pandas as pd
import csv
import time
# ここはご自分で発行されたKEYを入れてください
YOUR_API_KEY = 'b76272fc-acfd-d168-8005-0e38246ad871:fx'
# 翻訳したい入力テキスト
#TEXT='こんにちは。'
TEXT = '音を犠牲に全てを伝えられるなら、圧縮された別れの歌。'

params = {
            "auth_key": YOUR_API_KEY,
            "text": TEXT,
            "source_lang": 'JA', # 入力テキストの言語を日本語に設定（JPではなくJAなので注意）
            "target_lang": 'EN'  # 出力テキストの言語を英語に設定
        }
# パラメータと一緒にPOSTする
request = requests.post("https://api-free.deepl.com/v2/translate", data=params)

result = request.json()
#print(result)
print(result["translations"][0]["text"])
#
df=pd.read_csv("抽出.csv")

with open("/home/matsui/zemi/awesome-align/翻訳.csv",mode="w") as f:
    writer = csv.writer(f)
    writer.writerow(["出題文", "学習者","英語答え","翻訳" ])
    for ja,en,co in zip(df["出題文"],df["学習者"],df["英語答え"]):
        params = {
                    "auth_key": YOUR_API_KEY,
                    "text": ja,
                    "source_lang": 'JA', # 入力テキストの言語を日本語に設定（JPではなくJAなので注意）
                    "target_lang": 'EN'  # 出力テキストの言語を英語に設定
                }
        request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
        result = request.json()
        tr=result["translations"][0]["text"]
        writer.writerow([ja,en,co,tr])
        time.sleep(3)
