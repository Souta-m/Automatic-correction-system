import MeCab
import unidic
import nltk
import regex
from rest.modules.awesome_align import eijiro
from nltk.stem.wordnet import WordNetLemmatizer as WNL
wnl = WNL()  #単数形や原型にする機能
import requests
import time
import re
# 取り出したい品詞
#https://taku910.github.io/mecab/posid.html
select_conditions = ['動詞', '形容詞', '名詞','副詞',"感動詞","接続詞"]#訂正対象にする品詞
exclusion_mwords = ["ん","の","さ","こと","もの","よう","なく","なさい","せ"]#除外する単語1
exclusion_dwords = ["い","て","なっ","れる","し","られ","れ","する","あり","いる","あれ","なり","られる","ある","しまう"]#除外する単語2
exclusion_words = exclusion_mwords+exclusion_dwords
# 分かち書きオブジェクト
tagger = MeCab.Tagger("mecabのパス")
# 安定するらしい
tagger.parse('')

#形態素解析する時に起きる数字が分割される問題に対処するメソッド
def numbers(text):
    new_ja=[]
    node = tagger.parseToNode(text)
    tpas=[]
    while node:
        # 単語
        term = node.surface
        # 品詞
        pos = node.feature.split(',')[0]
        ano = node.feature.split(',')[1]
        if pos!="BOS/EOS":
            tpas.append([term,"",pos,ano])
        node = node.next
    list_term=[r[0] for r in tpas]+[" "]
    list_pos=[r[2] for r in tpas]+[" "]
    list_ano=[r[3] for r in tpas]+[" "]
    number_flag=False
    for i in range(len(list_pos)-1):
        if number_flag==True:
            if list_ano[i+1]!="数":
                number_flag=False
        elif list_ano[i]=="数" and list_ano[i+1]=="数":
            count=1
            number_flag=True
            new_term=list_term[i]
            while list_ano[i+count]=="数":
                new_term+=list_term[i+count]
                count+=1
            new_ja.append(new_term)
        else:
            new_ja.append(list_term[i])
    new_ja.append(list_term[-1])
    new_ja=" ".join(new_ja)
    new_ja=new_ja[:-2]
    return new_ja

#訳を増やすための翻訳メソッド
def translate(text):
    YOUR_API_KEY = 'あなたのAPIKEY'
    # 翻訳したい入力テキスト
    #TEXT='こんにちは。'
    params = {
                "auth_key": YOUR_API_KEY,
                "text": text,
                "source_lang": 'EN', # 入力テキストの言語を日本語に設定（JPではなくJAなので注意）
                "target_lang": 'JA'  # 出力テキストの言語を英語に設定
            }
    request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
    result = request.json()
    return result["translations"][0]["text"]

#以下漢数字とアラビア数字を対応付ける機能
tt_ksuji = str.maketrans('一二三四五六七八九〇壱弐参', '1234567890123')
re_suji = re.compile(r'[十拾百千万億兆\d]+')
re_kunit = re.compile(r'[十拾百千]|\d+')
re_manshin = re.compile(r'[万億兆]|[^万億兆]+')
TRANSUNIT = {'十': 10,
             '拾': 10,
             '百': 100,
             '千': 1000}
TRANSMANS = {'万': 10000,
             '億': 100000000,
             '兆': 1000000000000}
def kansuji2arabic(kstring: str, sep=False):
    """漢数字をアラビア数字に変換"""

    def _transvalue(sj: str, re_obj=re_kunit, transdic=TRANSUNIT):
        unit = 1
        result = 0
        for piece in reversed(re_obj.findall(sj)):
            if piece in transdic:
                if unit > 1:
                    result += unit
                unit = transdic[piece]
            else:
                val = int(piece) if piece.isdecimal() else _transvalue(piece)
                result += val * unit
                unit = 1

        if unit > 1:
            result += unit

        return result

    transuji = kstring.translate(tt_ksuji)
    for suji in sorted(set(re_suji.findall(transuji)), key=lambda s: len(s),
                           reverse=True):
        if not suji.isdecimal():
            arabic = _transvalue(suji, re_manshin, TRANSMANS)
            arabic = '{:,}'.format(arabic) if sep else str(arabic)
            transuji = transuji.replace(suji, arabic)
        elif sep and len(suji) > 3:
            transuji = transuji.replace(suji, '{:,}'.format(int(suji)))

    return transuji

#mecabの形態素解析で単語と品詞をdictで取り出すメソッド
def wakati_text(text,flag=False):

    # 分けてノードごとにする
    node = tagger.parseToNode(text)
    terms = []

    while node:

        # 単語
        term = node.surface

        # 品詞
        pos = node.feature.split(',')[0]
        #print(term,pos)

        # もし品詞が条件と一致してたら
        if pos in select_conditions and term not in exclusion_words:
            terms.append(term)

        node = node.next

    return terms

#mecabの形態素解析で単語と品詞をlistで取り出すメソッド
def pos_extra(text):

    node = tagger.parseToNode(text)
    dict1=[]
    dict2=[]
    while node:
        # 単語
        term = node.surface
        # 品詞
        pos = node.feature.split(',')[0]
        ano = node.feature.split(',')[1]
        if pos!="BOS/EOS":
            dict1.append([term,pos])
            dict2.append([term,ano])
        node = node.next
    #adjust(dict1,dict2)

    return dict1

#日本語文と英文中の単語の品詞を揃えるメソッド
def renketsu(tpas):  #動詞問題を解決する
    new_tpas=[]
    list_term=[r[0] for r in tpas]+[" "]
    list_pos=[r[2] for r in tpas]+[" "]
    list_ano=[r[3] for r in tpas]+[" "]
    list_ta=[r[4] for r in tpas]+[" "]
    for i in range(len(list_pos)-1):
        if list_pos[i]=="動詞" and list_ta[i]=="連用タ接続":
            new_tpas.append([list_term[i],list_term[i]+list_term[i+1],list_pos[i]])
        elif list_ano[i]=="サ変接続" and list_term[i+1] in ["し","する"]:
            count=1
            new_term=list_term[i]
            while list_term[i+count] in ["し","た","だ","する"]:
                new_term+=list_term[i+count]
                count+=1
            new_tpas.append([list_term[i],new_term,list_pos[i]])
        else:
            new_tpas.append([list_term[i],"",list_ano[i]])
    new_tpas.append([list_term[-1],"",list_ano[-1]])
    return new_tpas

#mecabの形態素解析で単語と品詞と活用形をlistで取り出すメソッド
def tpas_extra(text):

    node = tagger.parseToNode(text)
    tpas=[]
    poses=[]
    while node:
        # 単語
        term = node.surface
        # 品詞
        pos = node.feature.split(',')[0]
        ano = node.feature.split(',')[1]
        tasetsu = node.feature.split(",")[5]
        if pos!="BOS/EOS":
            poses.append([term,pos])
            tpas.append([term,"",pos,ano,tasetsu])
        node = node.next
    new_tpas=renketsu(tpas)
    # tpas=jyosi(tpas)
    return poses,new_tpas

#接頭詞をまとめたり、漢数字を対応させるためのメソッド
def kanji_setousi(word,lists):
    flag=True
    # print("＊間違え単候補")
    # print(word)
    # print(lists)
    for li in lists:
        if word[0]==li[0] and regex.findall(r'\p{Han}+',word[0])!=[]:
            flag=False
            break
        elif word[0] in ["お","ご"] and word[1:]==li:
            flag=False
            break
        elif bool(re.search(r'\d',li))==True: #数字
            new_word=kansuji2arabic(word)
            print("新しい数字",new_word)
            if new_word in li or li in new_word:
                flag=False
                break
            elif re.sub(r"\D", "", new_word)==re.sub(r"\D", "", li):
                flag=False
                break
        elif len(word)>=3 and word in li:
            flag=False
    return flag



#訂正対象とする単語を抽出するためのメソッド
def extra_teisei(lists,tgt2):
    
    teisei_words=[]
    for li in lists:
        if li[3] in select_conditions and li[0] not in exclusion_words and li[1].casefold()!=li[2].casefold() and li[2]!="" and li[1]!=li[2]: #助詞除く等の条件記述
            if li[1].isdecimal()==True: #li[1]に数字含むかどうか
                if li[0] not in li[1] and li[1] not in li[0]:
                    teisei_words.append([li[0],li[1],li[2]])
            elif "" == li[1]:
                jyukugo=eijiro.jyukugo(li[0])
                if len(jyukugo)==0:
                    teisei_words.append([li[0],li[1],li[2]])
                else:
                    teisei_jyukugo=li[2]
                    for jyu in jyukugo:
                        if jyu in tgt2:
                            teisei_jyukugo=jyu
                            break
                    teisei_words.append([li[0],li[1],teisei_jyukugo])
            elif " " in li[1]: #熟語の場合
                search_word=[li[1]]
                for l in li[1].split(" "):
                    search_word+=eijiro.search(wnl.lemmatize(l,pos="v"))+eijiro.search(wnl.lemmatize(l)) #英辞郎に記載されてあるかどうか
                search_word+=eijiro.search(li[1])
                search_word=list(filter(None, search_word))
                if len(search_word)<=3:  #Deepl翻訳の機能追加
                    search_word+=[translate(li[1])]
                if li[0] not in search_word:
                    teisei_words.append([li[0],li[1],li[2]])
            elif li[3]=="形容詞":
                search_word=eijiro.search(wnl.lemmatize(li[1],pos="a"))+eijiro.search(wnl.lemmatize(li[1]))+[li[1]]
                search_word=list(filter(None, search_word))
                if li[0] not in search_word and kanji_setousi(li[0],search_word)==True:
                    jyukugo=eijiro.jyukugo(li[0])
                    if len(jyukugo)==0:
                        teisei_words.append([li[0],li[1],li[2]])
                    else:
                        teisei_jyukugo=li[2]
                        for jyu in jyukugo:
                            if jyu in tgt2:
                                teisei_jyukugo=jyu
                                break
                        teisei_words.append([li[0],li[1],teisei_jyukugo])
                        
            else:
                search_word=eijiro.search(wnl.lemmatize(li[1],pos="v"))+eijiro.search(wnl.lemmatize(li[1]))+[li[1]]
                search_word=list(filter(None, search_word))
                if li[0] not in search_word and kanji_setousi(li[0],search_word)==True:
                    jyukugo=eijiro.jyukugo(li[0])
                    if len(jyukugo)==0:
                        teisei_words.append([li[0],li[1],li[2]])
                    else:
                        teisei_jyukugo=li[2]
                        for jyu in jyukugo:
                            if jyu in tgt2:
                                teisei_jyukugo=jyu
                                break
                        teisei_words.append([li[0],li[1],teisei_jyukugo])
    return teisei_words
