# 綱川研究室OB松井の引き継ぎ用資料のリポジトリです
修士論文および言語処理学会2023の資料を、[GoogleDrive](https://drive.google.com/drive/folders/1S-hn5aA6fnRFrR3Yy4Cj6Kg6GUHEPtO9?usp=sharing)にて公開しておきます。
松井の研究分野であった和文英訳問題自動添削や自動添削に関係する研究に興味がある方は、読んでみてもいいかもしれません。

[松井の提案システム](https://tsunalab.net/rest/system)は、止めていない限り公開してると思います。

## データやコーパスについて
データやコーパスは、容量が大きいため別のリンクにて保存しておきます。[ここ](https://drive.google.com/drive/folders/1S-hn5aA6fnRFrR3Yy4Cj6Kg6GUHEPtO9?usp=sharing)から必要なものをダウンロードしてください。
# コード説明
## 概要
綱川研究室松井颯汰の修士論文のコードです。

この研究を引継ぐにあたって、
①　python並びにDjangoを含めたの知識
②　自然言語処理における深層学習の知識
この二つ必要になってくるので、動かす前に簡単な勉強をお勧めします。
それぞれについて簡潔に説明します。
### ① CECtestフォルダ
Djangoの設定等を記述しているフォルダです。
### ② restフォルダ
提案システムのフレームワークであったり、処理部分を記述しているフォルダです。
### ③ staticフォルダ
cssファイルや画像ファイルがここに入ります。web公開する場合、別の場所に移動しなければなりませんが、ローカル環境であればここに記述して大丈夫です。
### ③ mange.py
Djangoでの実行は基本このmanage.pyで行います。
```
python manage.py runserver  #開発用サーバを立ち上げ
```
