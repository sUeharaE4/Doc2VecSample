# Doc2VecSample
Doc2Vecのフィージビリティチェックで環境構築したところ意外と面倒だったためdockerで動くようにしました。
あくまで動作確認レベルなのでチューニング等は一切していません。文章学習させて類似度が取れることだけ確認できるものです。
形態素解析するためのツールにMecab以外にもJumanがあるため、双方どちらも試せるようDockerfileは2つ作成しています。

## Mecab
おなじみMecabを形態素分解ツールとする版。Dockerfileをもとにbuildすればrondhuit社が公開しているlivedoor ニュースコーパスを学習します。
また、ローカル学習させた2019/04/30時点でのj-lyric人気アーティストTop100人の歌詞を学習させたモデルも同梱してます。「lyrics/アーティストID/歌詞.txt」のディレクトリ構成で
学習させているので、この構成でファイルを格納すれば歌詞を使って学習や推論ができます。
詳細はディレクトリ内のReadmeを参照してください。

## Juman
Juman++を形態素分解ツールとする版。RNNを使用した形態素解析器とのこと。
かなり容量を取るため、GCE無料枠での運用が難しそう(できなくはないと思う)。なのでサンプルプログラムも
<a href=https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563 target="_blank">参考サイト</a>
を大幅に参考にさせていただいている。gensimのバージョンが違うのか一部関数呼び出しの引数が変わっています。Buildすると17GB以上になるので要注意です。

### Django関連
もともと簡易的なWebサービス化する予定だったのでDjangoがありますが、こちらまで手を入れるかは未定です。