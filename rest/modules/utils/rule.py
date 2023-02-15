import nltk
not_pos=['NNP','NNPS','PRP','PRP$',',','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','JJR','JJS']
vrm_pos=['VB','VBD','VBG','VBN','VBP','VBZ','RB','MD']
vr_pos=['VB','VBD','VBG','VBN','VBP','VBZ','RB']
v_pos=['VB','VBD','VBP','VBZ']
j_pos=['JJ','JJR','JJS']
r_pos=['RB','RBR','RBS']
n_pos=['NN','NNS','NNP','NNPS']
nn_pos=['NN','NNS']
np_pos=['NNP','NNPS']
jaf_pos=['NN','NNS','NNP','NNPS','JJ','CC','VBG']
be_v=['is','are','am','was','were']
pp_s=['I','You','you','We','we','He','he','she','She','They','they']
thes=['The','the']

def Rule(string):
    pos_str=[]
    pos_wor=[]
    words = nltk.word_tokenize(string)
    pos=nltk.pos_tag(words)
    #print(pos)

    for p in pos:
        pos_str.append(p[1])
        pos_wor.append(p[0])
    for i,po in enumerate(pos_wor):
        if i==0:
            if po in be_v and pos_wor[-i]!='?':
                return False
            elif po in pp_s:
                if pos_str[1] not in vrm_pos:
                    print('not20')
                    return False
        elif i==len(pos_wor)-1:
            pass
        else:
            if po==pos_wor[i-1]:
                #print('not')
                return False
            elif pos_wor[i-1]=='the' and pos_wor[i+1]=='the':
                print('not19')
                return False
            #elif po in pp_s:
                #if pos_str[i+1] not in vrm_pos:
                    #print('not21')
                    #return False
            elif po =='?' or po=='.':
                return False


    for i,po in enumerate(pos_str):
        if i==0:
            if po=='PRP':
                if pos_str[i+1] not in vrm_pos:
                    #print('not3')
                    return False
            elif po in j_pos:
                if pos_str[i+1] not in jaf_pos:
                    #print('not4')
                    return False
            elif po in r_pos:
                if pos_str[i+1] in n_pos:
                    #print('not5')
                    return False
            elif po in np_pos:
                if pos_str[i+1] in nn_pos:
                    return False
            elif po in nn_pos:
                if pos_wor[i+1] in thes:
                    return False
        elif i==len(pos_str)-1:
            befpo=pos_str[i-1]
            if po in not_pos:
                if befpo==po:
                    #print('not2')
                    return False

        else:
            befpo=pos_str[i-1]
            if po in not_pos:
                if befpo==po:
                    #print('not1')
                    return False
            if po in j_pos:
                if pos_str[i+1] not in jaf_pos:
                    #print('not6')
                    return False
            elif po in r_pos:
                if pos_str[i+1] in n_pos:
                    #print('not7')
                    return False
            elif po in v_pos:
                if pos_str[i+1] in v_pos:
                    #print('not8')
                    return False
            elif po=="MD":
                if pos_str[i+1] not in vr_pos:
                    #print('not9')
                    return False
            elif po in np_pos:
                if pos_str[i+1] in nn_pos:
                    return False
            elif po in nn_pos:
                if pos_wor[i+1] in thes:
                    return False
    return True

def Sub(sent):
    sub=0
    s_words = nltk.word_tokenize(sent)
    s_wordp=nltk.pos_tag(s_words)
    for i,sw in enumerate(s_wordp):
        if i==0:
            pass
        else:
            if sw[1] in nn_pos and s_words[i-1] in nn_pos:
                sub=-0.01
    return sub


def First(ans,sent):
    sub=['I','You','We','They']
    be=['am','are','was','were']
    a_words = nltk.word_tokenize(ans)
    s_words=nltk.word_tokenize(sent)
    if a_words[0] in sub and s_words[0] in sub:
        a_words[0]=s_words[0]
        if a_words[1] in be and s_words[1] in be:
            a_words[1]=s_words[1]
    return " ".join(a_words)

def lastco(words):
    for i,word in enumerate(words):
        if word!='':
            if word[-1]=='.':
                words[i]=word[:-1]
    return words

def Fix(sentence):
    if sentence[0].isupper()==False:
        sentence=sentence[0].upper()+sentence[1:]
    for i in range(len(sentence)-1):
        if sentence[i]=='.' and sentence[i+1]=='.':
            sentence=sentence[:i]+sentence[i+1:]
    return sentence
#print(First('I am','You are'))
