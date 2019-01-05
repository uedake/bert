google-researchの[githubレポジトリのREADME](https://github.com/google-research/bert)の記載内容本文を翻訳したものです。

- 目次
 - [導入](#introduction%E5%B0%8E%E5%85%A5)
 - [BERTとは何か？](#what-is-bertbert%E3%81%A8%E3%81%AF%E4%BD%95%E3%81%8B)
 - [レポジトリで何が提供されているか？](#what-has-been-released-in-this-repository%E3%83%AC%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA%E3%81%A7%E4%BD%95%E3%81%8C%E6%8F%90%E4%BE%9B%E3%81%95%E3%82%8C%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B)
 - [事前学習済みモデル](#pre-trained-models%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%A2%E3%83%87%E3%83%AB)
 - [BERTを使った転移学習](#fine-tuning-with-bertbert%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E8%BB%A2%E7%A7%BB%E5%AD%A6%E7%BF%92)
 - [特徴ベクトルを抽出するためにBERTを使用する（Elmoのように）](#using-bert-to-extract-fixed-feature-vectors-like-elmo%E7%89%B9%E5%BE%B4%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E3%82%92%E6%8A%BD%E5%87%BA%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%ABbert%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%99%E3%82%8Belmo%E3%81%AE%E3%82%88%E3%81%86%E3%81%AB)
 - [トークン化](#tokenization%E3%83%88%E3%83%BC%E3%82%AF%E3%83%B3%E5%8C%96)
 - [BERTを使った事前学習](#pre-training-with-bertbert%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92)
 - [Colab上でBERTを使用する](#using-bert-in-colabcolab%E4%B8%8A%E3%81%A7bert%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%99%E3%82%8B)

---

# Introduction：導入

BERT(Bidirectional Encoder Representations from Transformers)は、広い範囲の自然言語処理タスクにおいて最先端（state-of-the-art）の結果を得る言語表現事前学習の新しい方法です。
BERTについての詳細及び数々のタスクの完全な結果は[学術論文](https://arxiv.org/abs/1810.04805)を参照ください。

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) 質問応答タスクの結果は下記です

|SQuAD v1.1 Leaderboard (Oct 8th 2018) |Test EM|Test F1|
|:--|--:|--:|
|1st Place Ensemble - BERT|87.4|93.2|
|2nd Place Ensemble - nlnet|86.0|91.7|
|1st Place Single Model - BERT|85.1|91.8|
|2nd Place Single Model - nlnet|83.5|90.1|

いくつかの自然言語推論タスクの結果は下記です

|System|MultiNLI|Question NLI|SWAG|
|:--|--:|--:|--:|
|BERT|86.7|91.1|86.3|
|OpenAI GPT (Prev. SOTA)|82.2|88.1|75.0|

その他多くのタスクでも効果的です。

さらに、これらの結果は、タスク特有のニューラルネットの設計をほぼすることなしに得られます。

もし、既にBERTが何かを知っておりただ利用を開始したいだけなら、[事前学習済みのモデルをダウンロード](#pre-trained-models%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%A2%E3%83%87%E3%83%AB)し、たった数分で最先端（state-of-the-art）の [転移学習を実行](#fine-tuning-with-bertbert%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E8%BB%A2%E7%A7%BB%E5%AD%A6%E7%BF%92)できます.

# What is BERT?：BERTとは何か？

BERTは言語表現事前学習の新しい方法です。その意味するところは、(Wikipediaのような)大きなテキストコーパスを用いて汎用目的の「言語理解」（language understanding）モデルを訓練すること、そしてそのモデルを関心のある実際の自然言語処理タスク(質問応答など)に適用することです。BERTは従来の方法を超えた性能を発揮します。なぜならBERTは、自然言語処理タスクを教師なしでかつ双方向に事前学習する初めてのシステムだからです。

教師なしとは、BERTが普通のテキストコーパスのみを用いて訓練されていることを意味します。これは、web上で莫大な量の普通のテキストデータが様々な言語で利用可能である為に、重要な特徴です。

事前学習済みの特徴表現は、文脈に依存する場合しない場合のいずれの場合もありえます。また、文脈に依存する特徴表現は、単方向である場合あるいは双方向である場合があり得ます. [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec)や[GloVe](https://nlp.stanford.edu/projects/glove/)のように文脈に依存しないモデルでは、語彙に含まれる各単語毎に「単語埋め込み」（word embedding）と呼ばれる特徴表現を生成します。よって、bankという単語はbank depositにおいてもriver bankにおいても同じ特徴表現となります。 一方で、文脈に依存するモデルでは、文に含まれる他の単語をもとにして各単語の特徴表現を生成します。

BERTは、文脈に依存する特徴表現の事前学習を行う最近の取り組みをもとに構築されました。それらの取り組みは、Semi-supervised Sequence Learning、 Generative Pre-Training、 ELMo、及びULMFitを含みます。しかし、それらの取り組みによるモデルは全て単方向もしくは浅い双方向でした。これは、各単語はただその左（あるいは右）に存在する単語によってのみ文脈の考慮がされることを意味します。例えば、`I made a bank deposit`という文では、`bank`の単方向特徴表現はただ`I made a`のみによって決まり、`deposit`は考慮されません。いくつかの前の取り組みでは、分離した左文脈モデルと右文脈モデルによる特徴表現を組み合わせていましたが、これは「浅い双方向」である方法です。BERTは、`bank`を左と右の両方の文脈`I made a ... deposit`をディープニューラルネットワークの最下層から用いて特徴表現します。その為、BERTは「深い双方向」（deeply bidirectional）なのです。

BERTは、シンプルなアプローチを用います：　入力された単語の15%をマスクする。シーケンス全体を深い双方向[Transformer encoder](https://arxiv.org/abs/1706.03762)に通す。そしてマスクされた単語だけを予測する。例えば、下記のように。

```
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

文の間の関係を学習するためには、任意の単言語コーパスから生成可能なシンプルなタスクを用いて訓練します。AとBの２つの文を与えられた時に、BがAの後にくる実際の文であるか、それともコーパス中のランダムな文であるかを判定するタスクです。

```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```

私たちは、大きなモデル(12層から24層のTransformer)を大きなコーパス(Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb))で長い時間をかけて(100万更新ステップ)訓練しました。それがBERTです。

BERTの利用は、事前学習と転移学習の２段階です。

**事前学習（pre-training）**はかなり高価です (4から16個のCloud TPUで4日 )が、各言語ごとに１回だけの手順です（現在のモデルでは英語のみですが、多言語モデルが近い将来リリースされます)。Googleによって事前学習された数多くの事前学習済みモデルをリリースしています。多くの自然言語処理の研究者は、一から自身のモデルを事前学習する必要はありません。

**転移学習（Fine-tuning）**は安価です。論文の全ての結果は、論文とまったく同じ事前学習済みモデルを用いて、一つのCloud TPUを用いればせいぜい１時間で、GPUを用いれば２、３時間で再現できます。 例えばSQuADは、一つのCloud TPUを用い30分で、１つのシステムとしては最先端（state-of-the-art）である91.0%のDev F1 スコアを達成できます。

BERTのもう１つの重要な側面は、多くの種類の自然言語処置タスクでとても簡単に採用できることです。論文の中で、文レベル (SST-2等)、文ペアレベル(MultiNLI等)、単語レベル (NER等)、スパンレベル(SQuAD等)のタスクについて、ほぼタスク特有の変更を行うことなしに、最先端の結果が得られることを示しています。

# What has been released in this repository?：レポジトリで何が提供されているか？

下記を提供しています：

- TensorFlow コード： BERTモデルアーキテクチャ構築の為(ほぼ標準的な[Transformer](https://arxiv.org/abs/1706.03762)のアーキテクチャ)
- 事前学習したチェックポイント：論文中のBERT-BaseとBERT-Largeそれぞれにつき、小文字化バージョンと大文字小文字混在バージョンの両方。
- TensorFlowコード： SQuAD、MultiNLI、MRPCを含む論文中で最も重要な転移学習の実験の、簡単にすぐ試せる実装（push-button replication）

このレポジトリの全てのコードはCPU、GPU、Cloud TPUで、（追加の設定なしに）すぐに実行できます。

# Pre-trained models：事前学習済みモデル

論文中のBERT-BaseモデルとBERT-Largeモデルを提供しています。小文字化（Uncased）は、テキストを小文字化してからWordPieceによるトークン化を用いることを意味します。例えば`John Smith`は`john smith`になります。小文字化（Uncased）モデルはアクセントマークの削除もします。大文字小文字混在（Cased）は、元のテキストの大文字小文字及びアクセントマークが維持されることを意味します。大文字小文字の情報が重要なタスク(例えば、固有名称認識、品詞のタグ付け)だと知っているのでない限り、典型的には小文字化（Uncased）モデルが優れています。

これらのモデルは全てソースコードとして同じライセンス(Apache 2.0)下で提供されます。

多言語モデルおよび中国語モデルについての情報は[多言語版のREADME](https://github.com/google-research/bert/blob/master/multilingual.md)を見てください。

大文字小文字混在（Cased）モデルを使用するには、`--do_lower=False`オプションを訓練用スクリプトに渡してください(もしくは、自分のスクリプトから使用する場合は、` do_lower_case=False`を`FullTokenizer`に直接渡してください。)

モデルへのリンクは下記です (右クリックし「保存」)：

- [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads, 110M parameters
- [BERT-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip): 24-layer, 1024-hidden, 16-heads, 340M parameters
- [BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads , 110M parameters
- [BERT-Large, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip): 24-layer, 1024-hidden, 16-heads, 340M parameters
- [BERT-Base, Multilingual Cased (New, recommended)](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
- [BERT-Base, Multilingual Uncased (Orig, not recommended) (Not recommended, use Multilingual Cased instead)](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
- [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

それぞれの.zipファイルは３つのアイテムを含みます：

- TensorFlow チェックポイント(`bert_model.ckpt`)：事前学習した重みを含む(実際には３ファイル).
- vocabファイル(`vocab.txt`)：WordPieceをword idへ紐づける為に使用
- 設定file (`bert_config.json`)：モデルのハイパーパタメータを指定する

# Fine-tuning with BERT：BERTを使った転移学習

**Important**: 論文の全ての結果は64GBのRAMを持つ一つのCloud TPUを使用して転移学習されています。現在のところ、12GBから16GBのRAMを有するGPUを用いた場合、BERT-Largeの論文中の結果の多くは再現することができません。なぜなら、メモリに収まる最大バッチサイズが小さすぎるからです。GPU上でより大きく効果的なバッチサイズを実現するためのコードをこのレポジトリに追加する作業を実施中です。詳細は、[メモリ不足の問題](#out-of-memory-issues%E3%83%A1%E3%83%A2%E3%83%AA%E4%B8%8D%E8%B6%B3%E3%81%AE%E5%95%8F%E9%A1%8C)のトピックを参照下さい。

このコードはTensorFlow 1.11.0で試験されています。Python2とPython3で試験されています(ただし、Google内部で使用されているPython2でより完全に試験されています)。

BERT-Baseを使用する転移学習の例は、与えらえたハイパーパラメータを用いて少なくとも12GBのRAMをもつGPUで実行すべきです。

## Fine-tuning with Cloud TPUs：Cloud TPUを使用した転移学習
下記のほとんどの例は、訓練／評価をTitan X or GTX 1080などのGPUを有するローカルマシンで実行することを念頭においています。

もし訓練を実行する為のCloud TPUの利用が可能なら、`run_classifier.py` もしくは`run_squad.py`を実行する際に下記のフラグを加えてください：

```
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

Cloud TPUの使い方については、[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)を見てください。代わりに、Google Colab notebookを使うこともできます（["BERT FineTuning with Cloud TPUs"](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)）。

Cloud TPU上では、事前学習済みモデルと出力先のディレクトリGoogle Cloud Storage上にある必要があります。例えば、`some_bucket`という名前のバケットをもっているなら、代わりに下記のフラグが使えます：

```
  --output_dir=gs://some_bucket/my_output_dir/
```

unzipされた事前学習済みモデルはGoogle Cloud Storageのフォルダ`gs://bert_models/2018_10_18`にあります。例えば：

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

## Sentence (and sentence-pair) classification tasks：文（あるいは文ペア）の分類タスク

この例を実行する前に、このスクリプトを実行しGLUEデータをダウンロードし、`$GLUE_DIR`に展開しなければいけません。次に、`BERT-Base`チェックポイントをダウンロードし、`$BERT_BASE_DIR`に展開しなければなりません。

このコード例では、BERT-BaseをMicrosoft Research Paraphrase Corpus (MRPC)コーパスで転移学習します。MRPCは3,600の例のみでなり、多くのGPUで2,3分で転移学習できます。

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
```

```
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

出力は下記のようになります

```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

開発用データセットにおける正解率（accuracy）は84.55%であったことを意味します。 たとえ、同じ事前学習済みチェックポイントから訓練を始めたとしても、MRPCのような小さなデータセットでは開発用データセットの正解率（accuracy）は大きな分散を持ちます。もし、何回か実行するなら(異なる出力先ディレクトリを指定することを忘れないように)、結果が84%から88%の間になることがわかるでしょう。

2,3の他の事前学習済みモデルがすぐに使える状態で`run_classifier.py`中に実装されています。あらゆる単一文もしくは文ペアの分類タスクでBERTを使うために、これらの例に従うことは簡単な方法です。

Note: `Running train on CPU`というメッセージを見るかもしれません。それは、単にCloud TPU以外の上で動作している（GPU上で動作していることを含む）を意味します。

### Prediction from classifier
一度分類器を訓練しさえすれば、`--do_predict=true`コマンドを利用することで推論モードの中で分類器を使用できます。 入力フォルダの中に`test.tsv`という名前のファイルが必要です。 出力フォルダ中の`test_results.tsv`という名前のファイルの中に出力が作成されます。 それぞれの行はサンプルを表し、列はそのクラスの確率です。

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/
```

## SQuAD 1.1

Stanford Question Answering Dataset (SQuAD)は人気のある質問応答ベンチマークのテストセットです。BERT (リリース時において)は、 SQuADにおいて、ほぼタスク特有の変更なしにかつデータ拡張なしに最も優れた（state-of-the-art）結果を獲得しました。semi-complexなデータ事前処理と事後処理を必要とします。それらの処理は (a) SQuAD における可変長のコンテキスト段落及び(b)SQuADの訓練で使用する文字レベルの回答アノテーション、に対処するための処理です。これらの処理は`run_squad.py`に実装及びドキュメント化されています。

SQuADを実行するには、最初にデータセットをダウンロードする必要があります。[SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/)は既にv1.1 データセットへのリンクがないようですが、必要なファイルは下記にあります。：

- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

これらを`$SQUAD_DIR`にダウンロードしてください。

現在のところ、12GBから16GBのRAMを有するGPUを用いた場合、メモリ制限により最も優れた論文中のSQuADの結果は再現することができません（実際、バッチサイズを1にしてさえ、BERT-Largeは12GBのGPU上にのりません）。しかし、十分に強力なBERT-Baseモデルが下記のパラメータを用いてGPU上で訓練できます。

```
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

開発用のデータセットの予測は、`output_dir`中の`predictions.json`という名前のファイルに保存されます:

```
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

下記のような結果になるはずです。

```
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

BERT-Baseでは、論文で報告されている88.5%に近い結果となります。

もしCloud TPUにアクセスできるなら、BERT-Largeを用いて訓練できます。下記は、SQuADのみで訓練されたsingle-systemとして約90.5%-91.0%のF1スコアを得るハイパーパラメータのセットです (少し論文と異なります) ：

```
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

例えば、このパラメータによるある１回の試行では、下記の結果になります:

```
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

この試行の前にTriviaQA上で1エポックの転移学習をするなら、結果はよくなります。ただし、あなたはTriviaQAをSQuAD jsonフォーマットに変換する必要があります。

## SQuAD 2.0
このモデルは`run_squad.py`に実装及びドキュメント化されています。

SQuAD 2.0上で実行するには、最初にデータセットのダウンロードが必要です。必要なファイルは下記にあります：

- train-v2.0.json
- dev-v2.0.json
- evaluate-v2.0.py

これらを`$SQUAD_DIR`にダウンロードしてください。

Cloud TPU上では、下記のようにBERT-Largeを用いて実行できます。：

```
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

出力ディレクトリから`./squad/.`という名前のローカルディレクトリに全てをコピーしていることを仮定します。初期の開発用データセットの予測は`./squad/predictions.json`にあります。無回答("")のスコアと非nullなベスト回答のスコアとの差異は`./squad/null_odds.json`ファイル中にあります。

「nullの予測」対「非null」間の閾値を調整するにはこのスクリプトを実行してください。

```
python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json ./squad/predictions.json --na-prob-file ./squad/null_odds.json
```

スクリプトの出力"best_f1_thresh"をTHRESHとしてください (典型的な値は-1.0から-5.0の間)。得られた閾値で予測を生成するためにモデルを再実行し、`./squad/nbest_predictions.json`からより適切な回答を抽出できます。

```
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
```

## Out-of-memory issues:メモリ不足の問題

論文中の全ての実験は64GBのRAMをもつCloud TPUを使って転移学習されています。 よって、論文に記載されたものと同じハイパーパラメータでは、12GBから16GBのRAMをもつGPUを使用する場合、メモリ不足の問題（out-of-memory issues）に遭遇するでしょう。

メモリ使用量に影響のある要因は:

- `max_seq_length`: 提供されたモデルは512までのシーケンス長（sequence length）で訓練されている。しかし、メモリを節約するために、より短い最大シーケンス長（sequence length）で転移学習してもよい。コード例中の`max_seq_length`フラグにより制御できる。
- `train_batch_size`: メモリ使用量は直接的にbatch sizeに比例する。
- モデルの種類 `BERT-Base` vs. `BERT-Large`： `BERT-Large`モデルは`BERT-Base`モデルよりかなり多くのメモリを必要とする。
- オプティマイザ: BERTのデフォルトのオプティマイザはAdamであり、`m`ベクトルと`v`ベクトルを保存するために多くの追加メモリを必要とする。メモリ効率の良いオプティマイザに切り替えれば、メモリ使用量は下がるが、結果に影響する。他のオプティマイザでは実験していない。

デフォルトの訓練用スクリプト(`run_classifier.py`及び`run_squad.py`)を使用し、TensorFlow 1.11.0を用いてTitan X GPU (12GB RAM)上での最大バッチサイズを計測した:

|System	Seq|Length|Max Batch Size|
|:--|--:|--:|
|BERT-Base|64|64|
|...|128|32|
|...|256|16|
|...|320|14|
|...|384|12|
|...|512|6|
|BERT-Large|64|12|
|...|128|6|
|...|256|2|
|...|320|1|
|...|384|0|
|...|512|0|

残念なことに、BERT-Large用の最大バッチサイズは、実際に正解率（accuracy）に悪影響を及ぼすほどに小さすぎる。GPU上でより大きく効果的なバッチサイズを実現するためのコードをこのレポジトリに追加する作業を実施中です。コードは、下記のいずれかもしくは両方のテクニックに基づきます：

- **Gradient accumulation**: ミニバッチ中のサンプルは、勾配の計算において互いに独立である(ここでは使用されていないバッチノーマライゼーションは除く)。複数のより小さなミニバッチの勾配は累積させた後に重みの更新を行ってもよく、これは単一の大きなバッチによる重みの更新と等価である。
- [Gradient checkpointing](https://github.com/openai/gradient-checkpointing): DNN訓練における主要なGPU/TPUメモリの使用は、フォワードパスにおける中間層の出力値のキャッシングである。キャッシュはバックワードパスの効率的な計算に必要である。"Gradient checkpointing"は賢い方法で中間層の出力値を再計算することで、計算時間を犠牲にする代わりにメモリ使用量を減らす。

**しかし、現在のリリースでは実装されていません。**

# Using BERT to extract fixed feature vectors (like ELMo)：特徴ベクトルを抽出するためにBERTを使用する（Elmoのように）

あるケースでは、転移学習よりも事前学習済みモデル全体が有益である。事前学習モデルの隠れ層が生成する値は、固定の文脈を考慮した各入力トークンの特徴表現であり「文脈を考慮して事前学習された単語埋め込み」が得られる。これはまたメモリ不足の問題のほとんどを和らげる。

例として、下記のように使用するスクリプト`extract_features.py`を同梱した：

```
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

これは、各Transformerから`layers`で指定する層のactivations情報を取り出し、JSONファイル(入力行ごとに１行) を生成します (`layers`に指定する値で-1は各Transformerの最後の隠れ層を指す)。

このスクリプトはとても大きな出力ファイルを生成することに注意(デフォルトでは、入力トークン毎に15kb)。
（訓練ラベルを反映するために）トークン化前後の間で対応関係を保持したいのであれば、下記のTokenization セクションを見てください。

**Note**: `Could not find trained model in model_dir: /tmp/tmpuB5g5c, running initialization to predict.`のようなメッセージを見かけるかもしれないが、このメッセージは予期されたものであり、単にsaved model APIでなく`init_from_checkpoint()` APIを使用していることを示しているだけである。チェックポイントを指定しないか、もしくは無効なチェックポイントを指定した場合、このスクリプトはエラーをだします。

# Tokenization：トークン化

文レベルのタスクもしくは文ペアのタスクの為に、トークン化はとてもシンプルです。単に`run_classifier.py`及び` extract_features.py`中のコード例に従えばよいです。文レベルのタスクの為の基本的な手順は下記です：

1. `tokenizer = tokenization.FullTokenizer(...)`：FullTokenizerをインスタンス化する
2. `tokens = tokenizer.tokenize(raw_text)`：テキストをトークン化する
3. 最大シーケンス長（sequence length）に切り捨てる(512までの値を使用可。メモリ使用量を抑えスピードを高速にする為に、可能であれば小さな値を使用したい)。
4. [CLS]トークンと[SEP]トークンを適切な位置に加える。

単語レベルやスパンレベルのタスク(例えば、SQuADやNER)ではより複雑になる。訓練ラベルを反映できるように、入力文と出力文の対応関係を保持する必要がある為。SQuADは、入力ラベルが文字ベースであるので、特に複雑な例である。SQuADの段落はしばしば我々の最大シーケンス長（sequence length）より長い。どのようにこの点を扱っているかは、`run_squad.py`中のコードを見てください。

単語レベルのタスクを扱う一般的なレシピを記述する前に、正確にtokenizerが何をしているか理解することは重要です。tokenizerには、３つの主要なステップがあります:

1. **テキストの正規化（Text normalization）**: 全ての空白文字をスペースに変換する。(小文字化モデルの場合)入力を小文字化しアクセントマークをはずす。例：`John Johanson's, → john johanson's,`
2. **区切り分割（Punctuation splitting）**: 区切り文字を２つに分割する (区切り文字の周りに空白を加える)。区切り文字は (a) P* Unicode classを持つ文字 (b) 英文字でなく数字でもスペースでもないASCII文字(例：技術的には区切りでないが、$のような文字). 例：` john johanson's, → john johanson ' s ,`
3. **WordPieceトークン化（WordPiece tokenization）**: 上記の手順の結果に空白トークン化を適用し、WordPieceトークン化を各tokenを別個に適用する(我々の実装は直接的にtensor2tensorの実装をベースにしている)。例：`john johanson ' s , → john johan ##son ' s ,`

この方法の優位性は多くの既存の英語tokenizerと互換性があること。例えば、下記のような品詞タグ付けタスクを思い浮かべて欲しい:

```
Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
```

トークン化された出力は下記になる：

```
Tokens: john johan ##son ' s house
```

重要なことに、まるでテキストが`John Johanson's house`（`'s`の前にスペースがない）であった場合と同じ出力となる。

もし、予めトークン化された特徴表現を単語レベルのアノテーションと共に有しているなら、シンプルに各入力単語をトークン化し 確定的にトークン化の前後での対応関係を維持してよい:

```
### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```

`orig_to_tok_map`は（トークン化前の文についている）ラベルをトークン化後の文に射影するために使用できる。

BERTが事前学習された方法と少しミスマッチを起こす一般的な英語トークン化方法があります。例えば、入力のトークン化が`do n't`のような縮約形を分割する場合にミスマッチを起こします。そうすることが可能であれば、事前にあなたのデータを生のテキストにみえるように戻す変換をするべきです。可能でないにしても、おそらく大きな影響はないでしょう。

# Pre-training with BERT：BERTを使った事前学習

任意のテキストコーパスにおいて"masked LM"及び"next sentence prediction"を行うためのコードを提供しています。これは論文に使用されたコード (オリジナルのコードはC++で書かれており、いくらか追加の複雑性を有しています)そのものではないですが、論文で記載の事前学習データを確かに生成します。

データ生成（data generation）を行う方法は以下の通りです。入力は通常のテキストファイルで、１行ごとに１文です("next sentence prediction"タスクの為には、次行に実際の「次の文」がくることが重要です)。ドキュメントは、空行で区切ってください。出力は`TFRecord`ファイルフォーマットにシリアライズされた`tf.train.Examples`のセットです。

spaCyのようなすぐに使える自然言語処理ツールキットを使用して、文のセグメンテーションを実行できます。` create_pretraining_data.py`スクリプトは、最大シーケンス長（sequence length）になるまでセグメントを連結します、それはパディングによる計算の無駄を最小化する為です(詳細はscriptを見てください)。しかし、あなたは入力にちょっとしたノイズを加えたいかもしれません(例えば、ランダムに2%の入力セグメントを切り捨てる等)、それは転移学習において、文法的に完全ではない（non-sentential）入力に対してもより頑健にする為の処置です。

スクリプトは全ての入力ファイル中の全ての事例をメモリ上に保存します、よって大きなデータファイルを使用する場合には、入力ファイルを分割した上でスクリプトを複数回実行する必要があります(`run_pretraining.py`にはファイルのglobを渡すこともできます。例えば、 `tf_examples.tf_record*`のように)

`max_predictions_per_seq`は、シーケンス毎のマスクされたLM予測の最大値です。この値は、およそ`max_seq_length` * `masked_lm_prob`ぐらいに設定すべきです(スクリプトは自動でこの値を設定しません、なぜなら正確な値を両方のスクリプトに渡す必要がある為）。

```
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

以下が事前学習の実行方法です。最初から事前学習を行いたい場合には、`init_checkpoint`をインクルードしないようにしてください。モデルの設定(vocab sizeを含む) は、`bert_config_file`中で指定します。このデモコードはただ小さなステップ(20)のみ事前学習しますが、実際には`num_train_steps`を10000ステップ以上に設定したいでしょう。`run_pretraining.py`に渡される`max_seq_length`と`max_predictions_per_seq`パラメータは、`create_pretraining_data.py`に渡される値と同じでなければなりません。

```
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

これは、下記の出力を得ます：

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

`sample_text.txt`ファイルはとても小さいのでこの訓練例は、たった2,3のステップの実行で過学習し、非現実的な高い正解率（accuracy）を叩き出すことに注意してください。

## Pre-training tips and caveats：事前学習のTipsと注意

- **もし、あなた自身が用意した語彙を使用したい場合、`bert_config.json`中の`vocab_size`の変更を忘れないようしてください。もし、この値を変更せずにより大きな語彙を使用すると、GPU or TPUで訓練するときに、境界値のチェックをしていない為にNaNsを得ることがあります。**
- もしタスクがその領域に特有の大きなコーパスが使用できる場合 (例えば"movie reviews"や"scientific papers"など)には、BERTのチェックポイントから始め、そのコーパスで追加の事前学習を行うことはおそらく有益でしょう。
- 論文で使用した学習率は、1e-4でした。しかし、BERTのチェックポイントから始めて、追加の事前学習をするなら、より小さな学習率(例えば、 2e-5)を使用すべきです。
- 現在のBERTモデルは英語のみですが、近い将来（願わくば2018年11月末に）多くの言語で事前学習された多言語版を提供する予定です。
より長いシーケンスの使用は、アテンション（attention）の処理にシーケンス長（sequence length）の２乗の計算量がかかることから、不相応に高価です。言い換えると、512の長さのシーケンスのサイズ64のバッチは、128の長さのシーケンスのサイズ256のバッチよりも大分高価です。全結合及び畳み込み層のコストも同様ですが、512のシーケンス長（sequence length）に対するアテンション（attention）のコストはとてつもなく大きいです。よって、ひとつの良いレシピは、例えば90,000ステップを128のシーケンス長で事前学習し、そして10,000の追加のステップを512のシーケンス長で事前学習することです。とても長いシーケンスはかなり高速に学ぶことができるpositional embeddingsを学習する為に必要でしょう。データを`max_seq_length`の異なる値で２回生成する必要があることに注意してください。
- 最初から事前学習したい場合、事前学習は計算量として高価であることに備えてください、特にGPUでは大変です。最初から事前学習したい場合、我々が推奨するレシピはBERT-Baseを単一の[preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing)で事前学習することです、それは凡そ2週間で、約$500 USDのコストがかかります(2018年10月時点の価格で)。単一のCloud TPU上で訓練するためには、論文中で使用した値と比べ、バッチサイズを小さくしなければならないかもしれません。TPUのメモリが許容する最大のバッチサイズを使用することをお勧めします。

## Pre-training data
我々は論文中で事前処理されたデータセットを提供することができません。Wikipediaによれば、推奨される事前処理は、最新のdumpをダウンロードし、`WikiExtractor.py`を使用してテキストを抽出し、任意の必要なデータ洗浄を実行し、普通のテキストに変換することです。

残念ながら、BookCorpusを収集した研究者は、既にそれを公にダウンロードできるようにしておりません。Guttenberg Datasetのプロジェクトは、パブリックドメインになっている古い本のいくらか小さい(200M単語)コレクションです。

Common Crawlは、テキストの大きなコレクションですが、BERTの事前学習をする為のコーパスを得るためには、かなりの事前処理とデータ洗浄をしなければならないでしょう。

## Learning a new WordPiece vocabulary

このレポジトリは新たなWordPiece語彙を学習するためのコードを含んでいません。その理由は論文で使われたコードはC++で実装され、Googleの内部ライブラリに依存している為です。英語については、ほぼ全ての場合、ただ我々の語彙を使用してモデルを事前学習するのがよいでしょうう。その他の言語の語彙を学習するには、たくさんのオープンソースの選択肢があります。しかし、我々の`tokenization.py`ライブラリとは互換性がないことに注意して下さい：

- Google's SentencePiece library
- tensor2tensor's WordPiece generation script
- Rico Sennrich's Byte Pair Encoding library

# Using BERT in Colab：Colab上でBERTを使用する

もしBERTを[Colab](https://colab.research.google.com/)上で使用したいなら、["BERT FineTuning with Cloud TPUs"](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)notebookから始められます。執筆時点 (2018年10月31日)で、ColabのユーザはCloud TPUに完全無料でアクセスできます。Note: １ユーザにつき１つ、使用可能時間は制限あり、ストレージ付きのGoogle Cloud Platformのアカウントが必要(ストレージは無料のクレジットで購入も可能)、将来も使用可能とは限らない。追加の情報は上記のリンクをクリック。

# FAQ

#### コードはCloud TPUと互換性がありますか？GPUはどうですか？
あります。このレポジトリの全てのコードは追加の調整なくCPU、GPU、Cloud TPUで動きます。ただしGPUによる訓練は単一のGPUのみです.

#### メモリ不足エラー（out-of-memory errors）になりました。何が悪いのでしょうか？
メモリ不足の問題のセクションを見てください。

#### PyTorch版はありますか？
公式のPyTorch版はありません。しかし、HuggingFaceの研究者がPyTorch版のBERTを用意しており、我々が事前学習したチェックポイントと互換性があり、結果の再現ができます。我々はPyTorch版の作成や維持にかかわっていませんので、質問は直接作者にコンタクトください。

#### Chainer版はありますか？
公式のChainer版はありません。しかし、Sosuke KobayashiがChainer版のBERTを用意しており、我々が事前学習したチェックポイントと互換性があり、結果の再現ができます。我々はChainer版の作成や維持にかかわっていませんので、質問は直接作者にコンタクトください。

#### 他の言語でのモデルは提供されますか？
はい、多言語版BERTモデルを近い将来提供します。どの言語が含まれるかについて約束はできませんが、十分なサイズのwikipediaデータを有する多くの言語を扱う単一モデルになる見込みです。

#### BERT-Largeより大きなモデルは提供されますか？
今のところ、BERT-Largeより大きなモデルの訓練は試みていません。十分な向上ができるならリリースするかもしれません。

#### 何のライセンスが適用されますか？
全てのコードとモデルは、Apache 2.0 license配下で提供されます。追加の情報はLICENSE fileを見てください。

#### どのようにBERTを引用できますか？
今のところ、Arxiv論文を引用してください。

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

論文をカンファレンスやジャーナルに投稿したらBibTeXを更新します。

