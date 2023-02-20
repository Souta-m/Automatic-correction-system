# 綱川研究室OB松井の引き継ぎ用資料のリポジトリです
修士論文および言語処理学会2023の資料を、[GoogleDrive](https://drive.google.com/drive/folders/1S-hn5aA6fnRFrR3Yy4Cj6Kg6GUHEPtO9?usp=sharing)にて公開しておきます。
松井の研究であった和文英訳問題自動添削や自動添削に関係する研究に興味がある方は、読んでみてもいいかもしれません。

[松井の提案システム](https://tsunalab.net/rest/system)は、止めていない限り公開してると思います。(https://tsunalab.net/rest/system)

## 研究プログラムを動かす前に
この研究を引継ぐにあたって<br>
①　pythonの知識<br>
②　Djangoの知識<br>
③　自然言語処理における深層学習の知識<br>
これら3つ必要になってくるので、動かす前に学習を強くお勧めします。これらの知識がないと動かすのは困難だと思います。以下にお勧めの勉強法紹介しておきます。
### ①　pythonの知識
pythonの知識に関しては、自然に身につくと思います。ネットも書籍も充実しているので、気に入ったもので独学で簡単に勉強してもらうのが一番良いと思います。<br>
pythonの学習が完了している人は、自然言語処理の学習も兼ねて、書籍等を参考にしつつテキスト分類などの深層学習モデルを構築してみるのも良いかなと思います。
### ②　Djangoの知識
Djangoとは、Webアプリケーションフレームワークのことです。深層学習を用いたWebアプリを構築する場合、Djangoを用いた開発が望ましいです。Django以外でのアプリケーション構築も可能だと思いますが、松井は最初PHPで実装しようとして躓きました。<br>
Djangoに関しては、ネットでの情報が充実していません。書籍で学びたい人は、Python Django 超入門(掌田 津耶乃著)などで学ぶことをお勧めします。Webで学びたい人は、[Djangoの公式ドキュメント](https://docs.djangoproject.com/ja/4.1/intro/)から学ぶことは可能です。
### ③　自然言語処理における深層学習の知識
自然言語処理分野は日々技術の進歩が目覚ましいので、松井が在籍していた時に使用していた技術や書籍が使えるかと言われると微妙です。最近の自然言語処理の動向を綱川先生に伺ったりネット等で調査することを強くお勧めします。深層学習以外にも、形態素解析等の自然言語処理技術の理解も必要です。<br>
一応、研究室にPythonで学ぶ自然言語処理系の書籍が多くあるので、まずはそれを触ってみて自然言語処理とpythonの理解を深めるところから始めるのもよいと思います。

## モデルやコーパスについて
モデルやコーパスは、容量が大きいため別のリンクにて保存しておきます。[ここ](https://drive.google.com/drive/folders/1S-hn5aA6fnRFrR3Yy4Cj6Kg6GUHEPtO9?usp=sharing)から必要なものをダウンロードしてください。

# コード説明
提案システムのコードについてです。
それぞれについて簡潔に説明します。フォルダに関しては、それぞれのフォルダに別のREADMEを用意しています。<br>
環境の違いやコード内にコメントしたことによるエラーが出現するかもしいことを承知ください。後、コード汚くてすみません……
## CECtestフォルダ
Djangoの設定等を記述しているフォルダです。
## restフォルダ
提案システムのフレームワークであったり、処理部分を記述しているフォルダです。
## staticフォルダ
静的ファイルや画像ファイルがここに入ります。web公開する場合は別の場所に移動しなければなりませんが、ローカル環境であればここに記述して大丈夫です。
## mange.py
Djangoでの実行は基本このmanage.pyで行います。manage.pyで何が出来るかは[ここ](https://qiita.com/okoppe8/items/7e3de8a4dd40b48debea)に書かれています。<br>
manage.pyとは少しずれますが、サーバー立ち上げ方法を以下に記載しておきます。
```
python manage.py runserver  #開発用サーバを立ち上げ(ローカル環境)
gunicorn --bind 127.0.0.1:8000 CECtest.wsgi:application  #本番用サーバ立ち上げ(グローバル環境)
```
## ファイル構造.txt
松井が実際に動かしていた環境下でのファイル構造が記載されています。
# 環境
・サーバー　hanamonoサーバー(詳しくは綱川先生にお聞きください)<br>
・OS　Linux 4.15.0-194-generic(Ubuntu 7.5.0)<br>
・エディタ　VScode<br>
# バージョン情報
・Python 3.7.13<br>
・nginx 1.14.0<br>
＊pythonのライブラリのバージョンはrequiremtns.txtを参照してください<br>
# 参考文献
・[DjangoでのWeb公開について](https://zenn.dev/hathle/books/django-vultr-book)<br>
提案システムをWeb公開する時にファイルに追記する等のいくつか設定が必要です。こちらの文献に詳しく書かれているので、この通りにすれば大丈夫だと思います。<br>
・[綱川研OB天野さんの研究](https://github.com/yuyaamano23/Hand_over_materials_NLP_LAB)<br>
私と同期だった天野さんの研究についてです。こちらも参考になると思うので、軽く読んでみるのをお勧めします。<br>
