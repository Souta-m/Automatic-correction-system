import nltk


def idiom_order(sents,teisei)

    for i,tei in enumerate(teisei):
        if " " in tei:
            indexes=[]
            numbert=0
            new_tei=[]
            tei_list=tei.split(" ")
            tei_len=len(tei_list)
            print(tei)
            for tl in range(tei_len):
                for i,sent in enumerate(sents_x):
                    if tei_list[tl]==sent:
                        indexes.append(i)
            zip_lists = zip(indexes,tei_list)
            print(zip_lists)
            zip_sort = sorted(zip_lists)
            print(zip_sort)
            indexes,tei_list=zip(*zip_sort)
            teisei[i]=" ".join(tei_list)
            
    return teisei

