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
YOUR_API_KEY = 'あなたが取得したAPI KEYを入力してください。'

def rest(request):
    return render(request, 'top.html')


class TopView(TemplateView):
    template_name = "top.html"

class HomeView(LoginRequiredMixin, TemplateView):
    template_name = "usage.html"

class LoginView(LoginView):
    form_class = forms.LoginForm
    template_name = "login.html"

class LogoutView(LoginRequiredMixin, LogoutView):
    template_name = "login.html"

def problem(request):
    return render(request,"problem.html")

def usage(request):
    return render(request,"usage.html")

def system(request):
    return render(request,"system.html")

def eval_system(request):
    return render(request,"experiment.html")

def my_customized_server_error(request, template_name='500.html'):
    return HttpResponseServerError('<h1>内部プログラムでエラーが出たので、松井まで連絡お願いします。</h1>')

def forecast(request):
    t1=time.time()
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
        request_deepl = requests.post("https://api-free.deepl.com/v2/translate", data=params)
        result = request_deepl.json()
        translate=result["translations"][0]["text"]



    #etext,flag=ginger.grammer_check(etext)
    # etext=etext.replace('[0m', '')
    # time_sta=time.perf_counter()
    # ans_string,cowords,miswords=pos.Mcec(result_estring,etext,[],[],0,True,time_sta)
    # miswords=rule.lastco(miswords)
    # ans_string=rule.Fix(ans_string)
    # data=[jtext,letext,ans_string]
    # return render(request, 'forecast.html', {'letext':letext,'result_estring':result_estring, 'etext': etext, 'ans_string': ans_string,
    # 'jtext': jtext,'cowords':cowords,'miswords':miswords,'flag':flag})
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
    if message =="誤りなし":
        return render(request,"output.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message})
    else:
        result=tensaku.teisei(jtext,etext,result_estring)
        if len(result)==0:
            message="誤りなし"
        print(time.time()-t1)
        return render(request,"output.html", {'jtext': jtext,'result_estring':result_estring, 'etext': etext,'message' : message,"result":result})


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

def evaluations(request):
    if request.POST['select']:
        print(request.POST['select'])
        with open("/home/matsui/DjangoPro/評価実験/各問題の添削スコア",mode="a") as f:
            f.write(request.POST['select']+"\n")
    return render(request,"experiment.html")


def final_evaluation(request):
    form1 = forms.FinalChoiceForm1()
    form2 = forms.FinalChoiceForm2()
    return render(request,"final_evaluation.html",{'form1':form1,'form2':form2})

def final_evaluation_write(request):
    print(request.POST)
    if request.POST['final_select1']:
        print(request.POST['final_select1'])
        with open("/home/matsui/DjangoPro/評価実験/最終的な添削のスコア.txt",mode="a") as f:
            f.write(request.POST['final_select1']+"\n")
    if request.POST['final_select2']:
        print(request.POST['final_select2'])
        with open("/home/matsui/DjangoPro/評価実験/提案システムは意味があったか.txt",mode="a") as f:
            f.write(request.POST['final_select2']+"\n")
    if request.POST['free_text']:
        print(request.POST['free_text'])
        with open("/home/matsui/DjangoPro/評価実験/自由記述.txt",mode="a") as f:
            f.write(request.POST['free_text']+"\n")
        with open("/home/matsui/DjangoPro/評価実験/各問題の添削スコア",mode="a") as f:
            f.write("\n")
    return render(request,"finish.html")
