# SimMch

マイクロホンアレイのシミュレーション収録を行うプロジェクト．
シングルチャネルの音源ファイル（wavファイル）からマイクロフォンアレイで収録したような多チャンネルのwavファイルを伝達関数，または幾何的な計算を利用して，生成する．
また，マイクロホンアレイを利用した音源定位や音源分離の基本的なアルゴリズムを実装．

# セットアップ
```
$ ./setup.sh
```

# 簡単な使い方
## シミュレーション

サンプルスクリプトファイル（./run_sample.sh）の中身

```
tf=./sample/tamago_rectf.zip
wav=./sample/jinsei_tanosii.wav
ch=0 # target channel of input wav files
dir=0 # degree

# generating sound
python ./sim_tf.py ${tf} ${wav} ${ch} ${dir} 1 ./sample/jinsei_tanosii_recons.wav
```

このスクリプトは，サンプルファイル（./sample/jinsei_tanosii.wav）から伝達関数（./sample/tamago_rectf.zip）を用いて0度方向からの音をシミュレートし，8chの音声ファイルである ./sample/jinsei_tanosii_recons.wav を生成する．

sim_tf.pyの引数

- 伝達関数ファイル
- 入力音ファイル
- 入力音ファイルのどのチャネルを使用するか（入力音ファイルが複数チャネルを持っていてもそのうち音源として使用できるのは１ｃｈのみ）
- 仮想的なマイクから音源がどの方向にあるかを指定
- スケールの値（通常は１でいいが，音量を変更したい場合などには1以下を，伝達関数によって音が小さくなってしまう場合には1以上を指定して調整する）
- 出力音ファイル（上書きされるので注意）


## 定位
### MUSIC 法

```
python music.py  ./sample/tamago_rectf.zip  ./sample/jinsei_tanosii.wav --out_npy test.npy --out_full_n
py test.npy --out_fig ./music.png --out_fig_with_bar ./music_bar.png --out_spectrogram ./fft.png 
```
## 分離
### ビームフォーミング

```
run_filters.sh
```

# そのほかのスクリプトファイル
- simmch.py
本プロジェクトで使われているユーティリティ関数
- make_noise.py
ホワイトノイズの音ファイルを作成する
- sim_td.py
時間差から多チャンネル音を合成する（伝達関数を使わない合成）
- check_parseval.py
パーセバルの等式が成り立っていることを確認する。（FFTのチェック用）

##　その他の機能（under construction） 
```
./make_sim_wav.sh ./i1.wav ./i2.wav ./o1.wav ./o2.wav
```
上のコマンドで
./i1.wav ./i2.wav
から混合音をシミュレーションで作成し
再分離した結果を
./o1.wav ./o2.wav
に保存される。
（デフォルトでは０度方向と９０度方向に音源がある場合のシミュレーションを行う）

# make_sim_wav.sh の中で使われている各種スクリプト

- wav_mix.py
複数の音ファイルを混合する
- const_sep.py
HARKを呼び出して、固定方向からの音を分離する


