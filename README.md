# 和文英訳問題自動添削における意味内容の添削を実施するシステム
[NLP2023のQ1-2の論文](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q1-2.pdf)のコード。<br>
[提案システム](https://tsunalab.net/rest/system)は、サーバが立ち上がっていれば公開中。(https://tsunalab.net/rest/system)


# モデルやコーパスについて
別途ダウンロードする必要あり。適宜自身で学習させたりダウンロードしたモデルでも可。

# Webでの公開方法
```
python manage.py runserver  #開発用サーバを立ち上げ(ローカル環境)
gunicorn --bind 127.0.0.1:8000 CECtest.wsgi:application  #本番用サーバ立ち上げ(グローバル環境)
```

# バージョン情報
・Python 3.7.13<br>
・nginx 1.14.0<br>
＊pythonのライブラリのバージョンはrequiremtns.txtを参照してください<br>
# 参考文献
・[和文英訳問題自動添削システムにおける意味内容の添削](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q1-2.pdf)<br>

