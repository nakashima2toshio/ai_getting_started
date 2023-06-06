## AI Getting Started

教科書：PyTorchチュートリアル（日本語翻訳版）https://yutaroogawa.github.io/pytorch_tutorials_jp/


参考書（１：本・Kindle）ニューラルネットワーク　実装ハンドブック
　https://www.amazon.co.jp/PyTorch%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF-%E5%AE%9F%E8%A3%85%E3%83%8F%E3%83%B3%E3%83%89%E3%83%96%E3%83%83%E3%82%AF-%E5%AE%AE%E6%9C%AC%E5%9C%AD%E4%B8%80%E9%83%8E-ebook/dp/B084JKYKMF/ref=sr_1_1?crid=2YTIMB8QMWZMP&keywords=pytorch%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E5%AE%9F%E8%A3%85%E3%83%8F%E3%83%B3%E3%83%89%E3%83%96%E3%83%83%E3%82%AF&qid=1686009023&sprefix=%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%80%80%E5%AE%9F%E8%A3%85%E3%83%8F%E3%83%B3%E3%83%89%E3%83%96%E3%83%83%E3%82%AF%2Caps%2C180&sr=8-1

参考書（２:Udemy）【E資格の前に】PyTorchで学ぶディープラーニング実装
 https://www.udemy.com/course/avilen-e-pytorch/


#### 1-(IDE) PyCharm professional
https://samuraism.com/2022/07/08/13654

## 目次

### データセットは以下から取得、利用した。

### Google Dataset Search
https://datasetsearch.research.google.com/

### Kaggle Dataset
https://www.kaggle.com/datasets

### 作成中・・・

| No. | 問題/問題名            | model 説明                                                                    |
| :-- | :--------------------- | :---------------------------------------------------------------------------- |
| 1   | 名称予測/IRIS          | MLP 分類問題：https://www.kaggle.com/datasets/aashimaaa/irisdataset           |
| 2_0 | 数字認識・分類/MNIST   | MLP 文字の分類:0～9までの手書き数字画像                                       |
| 2_1 | 画像分類/Fashion MNIST | CNN 画像の分類：10種類の「**ファッション商品**」写真の画像データセット7万枚。 |
| 2_2 | 画像分類/Fashion MNIST | CNN AlexNet                                                                   |
| 3   | 顔認識/Labeled Faces   | Face Recognition Dataset                                                      |
| 4   | 画像認識/CIFAR10       | CNN                                                                           |
| 5   | [チューニング手法]     | ハイパーパラメータ、モデルのチューニング                                      |
| 6   | 株価予測/Stock         | RNN                                                                           |
| 7   | 感情分析               | テキスト分類                                                                  |
| 8   | [構築済みモデルの利用] | Resnet18 iPhoneのnn                                                           |
| 9   | word2vector            | tokenizer                                                                     |
| 11  | LSTM                   | LSTM 超短期記憶                                                               |
| 12  | seq2seq                | transformer                                                                   |
| 13  | self-attention         | multi-head attention                                                          |
| 14  | 学習済BERT             | 学習済BERT                                                                    |
| 15  | BERT                   | BERT                                                                          |
| 16  | GPT-2                  | GPT-2                                                                         |
| 17  | GPT-3                  | GPT-3                                                                         |
| 18  | LLM  alpaca            | LLM  alpaca                                                                   |
| 19  | LLM                    | LLM                                                                           |


# 1.1「学習済みVGGモデル」
## 1.1.1「学習済みVGGモデル」を利用して画像分類を行う

