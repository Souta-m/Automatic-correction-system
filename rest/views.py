from django.shortcuts import render
import requests
import re
import csv
from transformers import *
from rest.modules import tensaku,bert_nli
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from . import forms
import time
YOUR_API_KEY = 'あなたが取得した機械翻訳システムのAPI KEYを入力してください。'

#topページはtop.htmlであることを宣言
def rest(request):
    return render(request, 'top.html')


#topページはtop.htnl
class TopView(TemplateView):
    template_name = "top.html"

#ホーム(ログイン後に遷移する画面)はusage.html
class HomeView(LoginRequiredMixin, TemplateView):
    template_name = "usage.html"

#ログインページはLogin.view
class LoginView(LoginView):
    form_class = forms.LoginForm
    template_name = "login.html"

#ログアウトもlogin.htmlで行う
class LogoutView(LoginRequiredMixin, LogoutView):
    template_name = "login.html"

#和文英訳問題を記載するページはproblem.html
def problem(request):
    return render(request,"problem.html")

#システムを説明するページはusage.html
def usage(request):
    return render(request,"usage.html")

#和文英訳自動添削システムはsystem.htnl
def system(request):
    return render(request,"system.html")

#評価実験用の和文英訳自動添削システムはexperiment.html
def eval_system(request):
    return render(request,"experiment.html")

#webアプリにエラーが出たらreturn内のメッセージを表示
def my_customized_server_error(request, template_name='500.html'):
    return HttpResponseServerError('<h1>内部プログラムでエラーが出たので、松井まで連絡お願いします。</h1>')

#自動添削後に遷移するoutput.html
def forecast(request):
    t1=time.time() #添削にかかった時間を計測するため
    jtext="" #出題文(日本語)
    etext="" #学習者訳(英語)
    cor_text="" #出題文の正解文(英語)
    #以下ブロック入力の確認
    if request.POST['jtext']:
        jtext = request.POST['jtext']
    if request.POST['cor_text']:
        cor_text = request.POST['cor_text']
    if request.POST['etext']:
        etext = request.POST['etext']
        letext = request.POST['etext']

    #必須入力が空白であったり、別言語で合った場合にエラーメッセージを出す。
    if jtext=="" or etext=="":
        error_message="必須項目を入力してください"
        print(error_message)
        return render(request, 'system.html',{'error_message': error_message})
    elif re.search(r'[ぁ-んァ-ン]', etext):
        error_message1="英訳部分には英語を入力してください"
        print(error_message1)
        return render(request, 'system.html',{'error_message1': error_message1})
    else:
        params = {
            "auth_key": YOUR_API_KEY,
            "text": jtext,
            "source_lang": 'JA', # 入力テキストの言語を日本語に設定（JPではなくJAなので注意）
            "target_lang": 'EN'  # 出力テキストの言語を英語に設定
        }
        request_deepl = requests.post("https://api-free.deepl.com/v2/translate", data=params) #翻訳API呼び出し
        result = request_deepl.json()
        translate=result["translations"][0]["text"]



    #以下cos類似度による正解文選択
    if cor_text=="":
        result_estring=translate
    else:
        cos1=tensaku.cos_sim(etext,cor_text)
        cos2=tensaku.cos_sim(etext,translate)
        if cos1>=cos2:
            result_estring=cor_text
        else:
            result_estring=translate

    message=tensaku.kensyutu(result_estring,etext) #誤り検出機能により、誤りなしかありかを判定。tensaku.pyのkensyutuメソッドを呼び出している。このような記述が今後多発する。
    if message =="誤りなし":
        return render(request,"output.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message}) #誤りがなかった場合、誤りなしと表示
    else:
        result=tensaku.teisei(jtext,etext,result_estring) #誤り訂正機能により、誤り単語等を抽出
        if len(result)==0:
            message="誤りなし" #訂正単語がない場合は誤りなしに変更
        print(time.time()-t1) #実行時間を端末上に表示
        return render(request,"output.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message,"result":result}) #誤り単語等を表示

#評価実験用の自動添削後に遷移するhtml。基本的に上のものと同じ
def experiments(request):
    jtext=""
    etext=""
    cor_text=""
    if request.POST['jtext']:
        jtext = request.POST['jtext']
    if request.POST['cor_text']:
        cor_text = request.POST['cor_text']
    if request.POST['etext']:
        etext = request.POST['etext']
        letext = request.POST['etext']

    if jtext=="" or etext=="":
        error_message="必須項目を入力してください"
        print(error_message)
        return render(request, 'experiment.html',{'error_message': error_message})

    elif re.search(r'[ぁ-んァ-ン]', etext):
        error_message1="英訳部分には英語を入力してください"
        print(error_message1)
        return render(request, 'experiment.html',{'error_message1': error_message1})
    else:
        params = {
            "auth_key": YOUR_API_KEY,
            "text": jtext,
            "source_lang": 'JA', # 入力テキストの言語を日本語に設定（JPではなくJAなので注意）
            "target_lang": 'EN'  # 出力テキストの言語を英語に設定
        }
        request_deepl = requests.post("https://api-free.deepl.com/v2/translate", data=params)
        result = request_deepl.json()
        translate=result["translations"][0]["text"]

    if cor_text=="":
        result_estring=translate
    else:
        cos1=tensaku.cos_sim(etext,cor_text)
        cos2=tensaku.cos_sim(etext,translate)
        if cos1>=cos2:
            result_estring=cor_text
        else:
            result_estring=translate
    message=tensaku.kensyutu(result_estring,etext)
    form = forms.SampleChoiceForm()
   
    if message =="誤りなし":
        return render(request,"evaluation.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message,'form':form})
    else:
        result=tensaku.teisei(jtext,etext,result_estring)
        if len(result)==0:
            message="誤りなし"
        return render(request,"evaluation.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message,"result":result,'form':form})

#添削一つごとの評価をファイルに格納(松井はtxtファイルで記録したが、csvでもデータベースでも可能)。終わったらexperiment.htmlに遷移するよう定義
def evaluations(request):
    if request.POST['select']:
        print(request.POST['select'])
        with open("評価実験用の記録しておきたい場所の自分のファイルパス",mode="a") as f:
            f.write(request.POST['select']+"\n")
    return render(request,"experiment.html")

#最終評価をするfinal_evaluation.html
def final_evaluation(request):
    form1 = forms.FinalChoiceForm1()
    form2 = forms.FinalChoiceForm2()
    return render(request,"final_evaluation.html",{'form1':form1,'form2':form2})

#最終評価をしたあと、同じように評価を格納。終わったらfinish.htmlに格納
def final_evaluation_write(request):
    print(request.POST)
    if request.POST['final_select1']:
        print(request.POST['final_select1'])
        with open("評価実験用の記録しておきたい場所の自分のファイルパス",mode="a") as f:
            f.write(request.POST['final_select1']+"\n")
    if request.POST['final_select2']:
        print(request.POST['final_select2'])
        with open("評価実験用の記録しておきたい場所の自分のファイルパス",mode="a") as f:
            f.write(request.POST['final_select2']+"\n")
    if request.POST['free_text']:
        print(request.POST['free_text'])
        with open("評価実験用の記録しておきたい場所の自分のファイルパス",mode="a") as f:
            f.write(request.POST['free_text']+"\n")
        with open("添削一つごとの評価を示すファイルパス,mode="a") as f:
            f.write("\n")
    return render(request,"finish.html")
