# 和文英訳問題自動添削における意味内容の添削を実施するシステム
[NLP2023のQ1-2の論文](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q1-2.pdf)のコード。<br>
[提案システム](https://tsunalab.net/rest/system)は、サーバが立ち上がっていれば公開中。(https://tsunalab.net/rest/system)<br>
＊一般公開中のため、機械翻訳APIの利用を止めています。
# 使用例
以下のように出題文とあなたの訳を入力してください(正解文1も必要)
![スクリーンショット 2023-03-17 203538](https://user-images.githubusercontent.com/82087359/225894284-9a1760e4-75af-4e09-b89d-e736902e40e6.png)
提出したら以下のようなフィードバックが返ってきます。
![スクリーンショット 2023-03-17 203607](https://user-images.githubusercontent.com/82087359/225894321-52567ddc-58ae-48f2-ab6a-39c8c07a395a.png)


# モデルやコーパスについて
別途ダウンロードする必要あり。適宜自身で学習させたりダウンロードしたモデルでも可。

# 端末上でのWebアプリ実行方法
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

