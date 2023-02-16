from rest.modules.awesome_align import pos
import nltk
import re
import re



#以下[日本語、正解、学習者訳]の単語を入れるコード
def conc_three(jap_words,x1s,x2s,jap_pos):
    concat_words=[]
    x1_count=0
    x2_count=0
    max_x1=len(x1s)-1
    max_x2=len(x2s)-1
    for i in range(len(jap_words)):
        #print(concat_words)
        if x1_count>max_x1 and x2_count>max_x2:
            concat_words.append([jap_words[i],"","",jap_pos[i]])
        elif x1_count>max_x1:
            concat_words.append([jap_words[i],x2s[x2_count][1],"",jap_pos[i]])
        elif x2_count>max_x2:
            concat_words.append([jap_words[i],"",x1s[x1_count][1],jap_pos[i]])
        elif jap_words[i]==x2s[x2_count][0] and jap_words[i]==x1s[x1_count][0]:
            concat_words.append([jap_words[i],x2s[x2_count][1],x1s[x1_count][1],jap_pos[i]])
            x1_count+=1
            x2_count+=1
        elif jap_words[i]==x2s[x2_count][0]:
            concat_words.append([jap_words[i],x2s[x2_count][1],"",jap_pos[i]])
            x2_count+=1
        elif jap_words[i]==x1s[x1_count][0]:
            concat_words.append([jap_words[i],"",x1s[x1_count][1],jap_pos[i]])
            x1_count+=1
        else:
            concat_words.append([jap_words[i],"","",jap_pos[i]])
    return concat_words  #出題文、学習者訳、正解文、品詞にしたい。
# concat_words=conc_three(jap_words,x1s,x2s)
# print(concat_words)
# leaner_words=[r[1] for r in concat_words]
# correct_words=[r[2] for r in concat_words]
# for l,c in zip(leaner_words,correct_words):
#     print(l,c)

#以下訂正単語で連結している単語をひとつのまとまりにするアルゴリズム
def jyukugo(teisei,jap_words,jap_poses,jap):
    teisei_count=0
    teisei_index=[]
    jyuku=[]
    def hantei(i,count,word,ans=""):
        #print(jap_words[i+1]+teisei[count+1])
        if jap_poses[i+1]=="助詞":
            return hantei(i+1,count,word,ans)
        elif jap_words[i+1]==teisei[count+1]:
            ans=re.findall(word+".*"+teisei[count+1],jap)
            if ans==[]:
                ans=[word+teisei[count+1]]
            return hantei(i+1,count+1,teisei[count+1],ans)

            #return i+1,count+1,word
        else:
            #print(teisei[count])
            return i+1,count+1,word,ans

    i=0
    while teisei[teisei_count]!="":
        if jap_words[i] == teisei[teisei_count]:
            i,teisei_count,word,ans=hantei(i,teisei_count,teisei[teisei_count])
            teisei_index.append(teisei_count)
            if isinstance(ans, list) and ans!=[]:
                jyuku.append(ans[0])
                #print(ans[0])
            else:
                jyuku.append(word)
                #print(word)

        else:
            i+=1
    return jyuku

def idiom_order(sents1,sents2,teiseis):
    def comma(tei,idx):
        for i in range(len(idx)-1):
            if idx[i]+1!=idx[i+1]:
                tei[i]=tei[i]+","
        return tei

    teisei=[r[1] for r in teiseis]
    for i,tei in enumerate(teisei):
        if " " in tei:
            indexes=[]
            numbert=0
            new_tei=[]
            tei_list=tei.split(" ")
            tei_len=len(tei_list)
            for tl in range(tei_len):
                for ii,sent in enumerate(sents1):
                    if tei_list[tl]==sent:
                        indexes.append(ii)
            zip_lists = zip(indexes,tei_list)
            zip_sort = sorted(zip_lists)
            indexes,tei_list=zip(*zip_sort)
            tei_list=list(tei_list)
            tei_list=comma(tei_list,indexes)
            teiseis[i][1]=" ".join(tei_list)
    
    teisei=[r[2] for r in teiseis]
    for i,tei in enumerate(teisei):
        if " " in tei:
            indexes=[]
            numbert=0
            new_tei=[]
            tei_list=tei.split(" ")
            tei_len=len(tei_list)
            for tl in range(tei_len):
                for ii,sent in enumerate(sents2):
                    if tei_list[tl]==sent:
                        indexes.append(ii)
            zip_lists = zip(indexes,tei_list)
            zip_sort = sorted(zip_lists)
            indexes,tei_list=zip(*zip_sort)
            tei_list=list(tei_list)
            tei_list=comma(tei_list,indexes)
            teiseis[i][2]=" ".join(tei_list)
            
    return teiseis

