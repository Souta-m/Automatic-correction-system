# rest/modeulesフォルダの説明です
modulesフォルダには、研究で提案した誤り検出と誤り訂正について処理するためのpythonファイルが格納されています。<br>
＊私(松井)が見てもかなり汚いコードだと思います。本当に申し訳ございません。
## awesome-alignフォルダ
誤り訂正で用いる単語アライメントモデルを動かし、訂正単語等の情報を取得するためのフォルダです。このフォルダも別途READMEを用意しておきます。
## utilsフォルダ
BERTによる自然言語推論モデルを動かすためのPythonファイルが入っているフォルダです。自然言語推論を用いて誤り検出する場合特に変更はないですが、BERT以外のモデルやそもそも自然言語推論を用いず誤り検出するなら消去してもらって大丈夫です。
## bert_nli.py
BERTによる自然言語推論モデルを定義しています。
## tensaku.py
上で説明してきた機能をまとめているPythonファイルです。このPythonファイル内のメソッドをviews.pyで呼び出し、Djangoでの添削アプリを実現しています。
#参考文献
[Awesome-align](https://github.com/neulab/awesome-align)<br>
[BERT_NLI](https://github.com/yg211/bert_nli)<br>
[Sentence-BERT](https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part9.html)<br>
