## XXXX 畳み込みニューラルネットワーク 画像分類プログラム

1. 前準備
2. 学習・訓練データとテストデータの用意

   1. データセットの読み込み
   2. データの中身を確認
   3. ミニバッチデータセットの確認
   4. ミニバッチサイズを指定したデータローダーを作成
   5. 学習済みのニューラルネットワークの読み込み
      1. CPUとGPUどちらを使うかを指定
      2. ニューラルネットワークのパラメータが更新されないようにする
      3. 損失関数と最適化関数の定義
3. 学習　---------学習パートはここから---------

   1. 損失と正解率を保存するリストを作成
   2. 学習（エポック）の実行
      1. エポックの進行状況を表示
   3. 損失と正解率の初期化
   4. ニューラルネットワークを学習モードに設定
   5. ミニバッチごとにデータをロードし学習
   6. GPUにTensorを転送
   7. 勾配を初期化
   8. データを入力して予測値を計算（順伝播）
   9. 損失（誤差）を計算
   10. 勾配の計算（逆伝搬）
   11. パラメータ（重み）の更新
   12. ミニバッチごとの損失を蓄積
   13. 予測したラベルを予測確率y_pred_probから計算
   14. ミニバッチごとに正解したラベル数をカウント
   15. エポックごとの損失と正解率を計算（ミニバッチの平均の損失と正解率を計算）
4. ---------学習パートはここまで---------
5. CPUとGPUどちらを使うかを指定
6. 学習済みのAlexNetを取得
7. ニューラルネットワークのパラメータが更新されないようにする
8. 損失関数と最適化関数の定義

## XXXX データセットの転移学習【コード作成の手順】

## 1. 前準備（パッケージのインポート

### 必要なパッケージのインストール

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
```

## 4.2. 訓練データとテストデータの用意

### CIFAR10データセットの読み込み

```python
train_dataset = torchvision.datasets.CIFAR10(
    root='./data/',     # データの保存場所
    train=True,         # 学習データかどうか
    download=True,      # ダウンロードするかどうか
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],  # RGBの平均
            [0.5, 0.5, 0.5],  # RGBの標準偏差
            )
    ])
)
```

```python
test_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],    # RGBの平均
            [0.5, 0.5, 0.5],  # RGBの標準偏差
            )
        ])
)
```

### train_datasetの中身を確認

image, label = train_dataset[0]
print("image size: {}".format(image.size()))  # 画像サイズ
print("label: {}".format(label))  # ラベルサイズ

### ミニバッチサイズを指定したデータローダーを作成

```python
train_batch = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=64,
shuffle=True,
num_workers=2)
test_batch = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=64,
shuffle=False,
num_workers=2)
```

### ミニバッチデータセットの確認

for images, labels in train_batch:
print("batch images size: {}".format(images.size()))  # バッチの画像サイズ
print("image size: {}".format(images[0].size()))  # 1枚の画像サイズ
print("batch labels size: {}".format(labels.size()))  # バッチのラベルサイズ
break
#%% md

## 4.3. 学習済みのニューラルネットワークの読み込み

### CPUとGPUどちらを使うかを指定

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

### 学習済みのAlexNetを取得

"""
net = models.alexnet(pretrained=True)
net = net.to(device)
print(net)  # AlexNetの構造を表示

# ニューラルネットワークのパラメータが更新されないようにする

for param in net.parameters():
param.requires_grad = False
net = net.to(device)
#%%
#出力層の出力を1000クラス用から10クラス用に変更
num_features = net.classifier[6].in_features  # 出力層の入力サイズ
num_classes = 10  # CIFAR10のクラスの数を指定
net.classifier[6] = nn.Linear(num_features, num_classes).to(device)  # 出力を1000から2へ変更

print(net)
#%% md

## 4.4. 損失関数と最適化関数の定義

#%%

# 損失関数の定義

criterion = nn.CrossEntropyLoss()

# 最適化関数の定義

optimizer = optim.Adam(net.parameters())
#%% md

## 4.5. 学習

#%%

# 損失と正解率を保存するリストを作成

train_loss_list = []  # 学習損失
train_accuracy_list = []  # 学習データの正答率
test_loss_list = []  # 評価損失
test_accuracy_list = []  # テストデータの正答率

# 学習（エポック）の実行

epoch = 10
for i in range(epoch):

# エポックの進行状況を表示

print('---------------------------------------------')
print("Epoch: {}/{}".format(i+1, epoch))

# 損失と正解率の初期化

train_loss = 0  # 学習損失
train_accuracy = 0  # 学習データの正答数
test_loss = 0  # 評価損失
test_accuracy = 0  # テストデータの正答数

## ---------学習パート---------

### ニューラルネットワークを学習モードに設定

net.train()

### ミニバッチごとにデータをロードし学習

for images, labels in train_batch:

### GPUにTensorを転送

images = images.to(device)
labels = labels.to(device)

### 勾配を初期化

optimizer.zero_grad()

### データを入力して予測値を計算（順伝播）

y_pred_prob = net(images)

### 損失（誤差）を計算

loss = criterion(y_pred_prob, labels)

### 勾配の計算（逆伝搬）

loss.backward()

### パラメータ（重み）の更新

optimizer.step()

### ミニバッチごとの損失を蓄積

train_loss += loss.item()

### 予測したラベルを予測確率y_pred_probから計算

y_pred_labels = torch.max(y_pred_prob, 1)[1]

### ミニバッチごとに正解したラベル数をカウント

train_accuracy += torch.sum(y_pred_labels == labels).item() / len(labels)

### エポックごとの損失と正解率を計算（ミニバッチの平均の損失と正解率を計算）

epoch_train_loss = train_loss / len(train_batch)
epoch_train_accuracy = train_accuracy / len(train_batch)

# ---------学習パートはここまで---------

# ---------評価パート---------

# ニューラルネットワークを評価モードに設定

net.eval()

# 評価時の計算で自動微分機能をオフにする

with torch.no_grad():
for images, labels in test_batch:

### GPUにTensorを転送

images = images.to(device)
labels = labels.to(device)

### データを入力して予測値を計算（順伝播）

y_pred_prob = net(images)

### 損失（誤差）を計算

loss = criterion(y_pred_prob, labels)

### ミニバッチごとの損失を蓄積

test_loss += loss.item()

# 予測したラベルを予測確率y_pred_probから計算

y_pred_labels = torch.max(y_pred_prob, 1)[1]

# ミニバッチごとに正解したラベル数をカウント

test_accuracy += torch.sum(y_pred_labels == labels).item() / len(labels)

### エポックごとの損失と正解率を計算（ミニバッチの平均の損失と正解率を計算）

```python
epoch_test_loss = test_loss / len(test_batch)
epoch_test_accuracy = test_accuracy / len(test_batch)
```

##### ---------評価パートはここまで---------

### エポックごとに損失と正解率を表示

print("Train_Loss: {:.4f}, Train_Accuracy: {:.4f}".format(epoch_train_loss, epoch_train_accuracy))
print("Test_Loss: {:.4f}, Test_Accuracy: {:.4f}".format(epoch_test_loss, epoch_test_accuracy))

### 損失と正解率をリスト化して保存

train_loss_list.append(epoch_train_loss)
train_accuracy_list.append(epoch_train_accuracy)

test_loss_list.append(epoch_test_loss)
test_accuracy_list.append(epoch_test_accuracy)

## 4.6. 結果の可視化

### 損失

```python:plot_loss
plt.figure()
plt.title('Train and Test Loss')       # タイトル
plt.xlabel('Epoch')                    # 横軸名
plt.ylabel('Loss')                     # 縦軸名
plt.plot(range(1, epoch+1), train_loss_list, color='blue',linestyle='-', label='Train_Loss')  # Train_lossのプロット
plt.plot(range(1, epoch+1), test_loss_list, color='red',linestyle='--', label='Test_Loss')  # Test_lossのプロット
plt.legend()                           # 凡例
```

### 正解率

```python:plot_accuracy
plt.figure()
plt.title('Train and Test Accuracy')     # タイトル
plt.xlabel('Epoch')                      # 横軸名
plt.ylabel('Accuracy')                   # 縦軸名
plt.plot(range(1, epoch+1), train_accuracy_list, color='blue',linestyle='-', label='Train_Accuracy')  # Train_lossのプロット
plt.plot(range(1, epoch+1), test_accuracy_list, color='red',linestyle='--', label='Test_Accuracy')    # Test_lossのプロット
plt.legend()
```

# 表示

plt.show()
