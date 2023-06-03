## メニュー


| No. | Model    | Description                                                                                                                                                                                                        |
| --- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | 回帰問題 | IRIS:"setosa"、"versicolor"、"virginica"と呼ばれる3種類の品種のアヤメがあります。<br />このアヤメの花冠（はなびら全体のこと）を表すデータとして、<br/>がく片（Sepal）、花弁（Petal）の<br />幅及び長さがあります。 |
| 2   | RNN      | MNIST: 文字の分類                                                                                                                                                                                                  |
| 3   | RNN      | MNIST: 文字の分類                                                                                                                                                                                                  |

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
4. 学習（エポック）の実行
   1. エポックの数だけ
      1. エポックの進行状況を表示
      2. 損失の初期化
         train_loss = 0  # 学習損失（MSE）
         test_loss = 0   # 評価損失（MSE）
         train_mae = 0   # 学習MAE
         test_mae = 0    # 評価MAE# ---------学習パート---------

         # ニューラルネットワークを学習モードに設定

         # ミニバッチごとにデータをロードし学習

         for data, label in train_batch:

         # GPUにTensorを転送

         # 勾配を初期化

         # データを入力して予測値を計算（順伝播）

         # 損失（誤差）を計算

         # 勾配の計算（逆伝搬）

         # パラメータ（重み）の更新

         # ミニバッチごとの損失を蓄積# ミニバッチの平均の損失を計算

         batch_train_loss = train_loss / len(train_batch)
         batch_train_mae = train_mae / len(train_batch)
5. -学習パートはここまで---------
6. -評価パート---------
   1. ニューラルネットワークを評価モードに設定
      net.eval()

      # 評価時の計算で自動微分機能をオフにする

      with torch.no_grad():
      for data, label in test_batch:

      # GPUにTensorを転送

      data = data.to(device)
      label = label.to(device)

      # データを入力して予測値を計算（順伝播）※　

      y_pred = net(data)

      # 損失（誤差）を計算

      loss = criterion(y_pred, label)
      mae = criterion2(y_pred, label)

      # ミニバッチごとの損失を蓄積

      test_loss += loss.item()
      test_mae += mae.item()

      # ミニバッチの平均の損失を計算

      batch_test_loss = test_loss / len(test_batch)
      batch_test_mae = test_mae / len(test_batch)

      # ---------評価パートはここまで---------

      # エポックごとに損失を表示

      print("Train_Loss: {:.4f} Train_MAE: {:.4f}".format(batch_train_loss, batch_train_mae))
      print("Test_Loss: {:.4f} Test_MAE: {:.4f}".format(batch_test_loss, batch_test_mae))

      # 損失をリスト化して保存

      train_loss_list.append(batch_train_loss)
      test_loss_list.append(batch_test_loss)
      train_mae_list.append(batch_train_mae)
      test_mae_list.append(batch_test_mae)

## 4.6. 結果の可視化

### 損失

```python:plot_loss
# タイトル
# 横軸名
# 縦軸名
# Train_lossのプロット
# Test_lossのプロット
# 凡例
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
