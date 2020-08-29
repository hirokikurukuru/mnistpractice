import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop
img_width = 28 
img_height = 28 
num_input = int(img_width * img_height)
(X_train,y_train),(X_test,y_test)= mnist.load_data()
X_train  = X_train.reshape(60000, num_input)
X_test   = X_test.reshape(10000, num_input)
X_train  = X_train.astype('float32')
X_test   = X_test.astype('float32')
X_train /= 255
X_test  /= 255
y_train  = keras.utils.to_categorical(y_train, 10)
y_test   = keras.utils.to_categorical(y_test, 10)
def main():
     # データセットの個数を表示,訓練データもテストデータも一次元化したのでshape[0]でaxis=0について表示すればいい
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # ハイパーパラメータ
    batch_size = 128 # バッチサイズ
    num_classes = 10 # 分類クラス数(今回は0～9の手書き文字なので10)
    epochs = 20      # エポック数(学習の繰り返し回数)
    dropout_rate = 0.2 # 過学習防止用：入力の20%を0にする（破棄）
    num_middle_unit = 512 # 中間層のユニット数
    #モデルの構築
    model = Sequential()
    # 4層のMLP(多層パーセプトロン)のモデルを設定
    # 1層目の入力層:28×28=784個のユニット数
    # 2層目の中間層:ユニット数512、活性化関数はrelu関数
    model.add(Dense(activation='relu', input_dim=num_input, units=num_middle_unit))
    # ドロップアウト(過学習防止用, dropout_rate=0.2なら512個のユニットのうち、20%のユニットを無効化）
    model.add(Dropout(dropout_rate)) # 過学習防止用
     # 4層目の出力層:10分類（0から9まで）なので、ユニット数10, 分類問題なので活性化関数はsoftmax関数(全ユニットからの出力の合計が位1.0)
    model.add(Dense(num_classes, activation='softmax')) # 活性化関数：softmax
    model.summary()
     # コンパイル（多クラス分類問題、損失関数はcategorical_crossentropy）
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    # 構築したモデルで学習（学習データ:trainのうち、10％を検証データ:validationとして使用）
    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_split=0.1)
    #verbose: 0または1．進行状況の表示モード．0 = 表示なし，1 = プログレスバー
    # テスト用データセットで学習済分類器に入力し、パフォーマンスを計測
    score = model.evaluate(X_test, 
                            y_test,
                            verbose=0
                            )
    # パフォーマンス計測の結果を表示
    # 損失値（値が小さいほど良い）
    print('Test loss:', score[0])

    # 正答率（値が大きいほど良い）
    print('Test accuracy:', score[1])
main()

