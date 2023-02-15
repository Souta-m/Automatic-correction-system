import nltk
from nltk.corpus import wordnet as wn

def create_mask_set(sent,swa):
    sentences = []
    flags=[]
    sent = sent.strip().split()
    pos=nltk.pos_tag(sent)
    for i in range(len(sent)):
        if ['MD']== swa:
            if pos[i][1] in ['VBD','VBP','VBZ']:
                new_sent = sent[:]
                #print(new_sent)
                new_sent[i]=wn.morphy(new_sent[i])
                new_sent.insert(i, '[MASK]')
                #print(new_sent)
                new_sent = filter(None, new_sent)
                text = " ".join(new_sent)
                sentences.append(text)
                flags.append(0)
        else:
            if i==0:
                new_sent = sent[:]
                new_sent.insert(i, '[MASK]')
                text = " ".join(new_sent)
                sentences.append(text)
                flags.append(0)

            elif i==len(sent)-1:
                new_sent = sent[:]
                new_sent.insert(len(sent), '[MASK]')
                text = " ".join(new_sent)
                sentences.append(text)
                flags.append(0)
            if pos[i][1] in swa:
                if i==0:
                    new_sent = sent[:]
                    new_sent[i] = '[MASK]'
                    text = " ".join(new_sent)
                    sentences.append(text)
                    flags.append(1)
                elif i==len(sent)-1:
                    new_sent = sent[:]
                    new_sent[i] = '[MASK]'
                    text = " ".join(new_sent)
                    sentences.append(text)
                    flags.append(1)
                else:
                    new_sent = sent[:]
                    new_sent[i] = '[MASK]'
                    text = " ".join(new_sent)
                    sentences.append(text)
                    flags.append(1)
                    new_sent=sent[:]
                    new_sent.insert(i, '[MASK]')
                    text1=" ".join(new_sent)
                    sentences.append(text1)
                    flags.append(0)
                    new_sent=sent[:]
                    new_sent.insert(i+1, '[MASK]')
                    text2=" ".join(new_sent)
                    sentences.append(text2)
                    flags.append(0)

    return sentences,flags
