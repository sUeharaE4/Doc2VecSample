# Doc2VecSample Juman
JUMAN++, gensimをエンジンとしたDoc2Vecのサンプル。Buildすると17GB以上のサイズになるのでGCP無料枠での運用は厳しいかなと思い、雑なつくりです。。。

## Build
docker-compose build
  
を実行すればOKです。必要なものはインストールして、すでに学習済みのモデルを使って類似度を返すまでがbuildに含まれます。
build後はコンテナに入ってsample.pyを叩いてください。後述しますが、docker-compose up しても何も起こりません。。  
docker imagesでイメージID確認して、 docker run -it イメージID bash でコンテナに入って叩いてください(不便ですみませんが、あくまで動作確認なので。)

## 学習データのディレクトリ構成
Dockerfileをもとにbuildすればrondhuit社が公開しているlivedoor ニュースコーパスを学習するようになっています。
(デフォルトで学習済みモデルを参照して推論するようになっているので、学習が目的であればモデルを削除してください)
SAMPLE_TEXT_ROOT_DIR  
独自データを学習させたければ「pythonScripts」ディレクトリにデータを格納して再度buildすればデータがコンテナに入ります。sample.pyのSAMPLE_TEXT_ROOT_DIRを
学習データのルートディレクトリを指定すれば配下のファイルを再帰的に取得して学習対象にします。PRE_TRAIN_MODEL_PATHは学習済みモデルの保存先
だけでなく、指定したファイルがあれば学習せず推論するルートに進むので学習させたい場合はファイルを削除しておくか、別のパスを指定してください。

### Django関連
もともと簡易的なWebサービス化する予定だったのでDjangoがありますが、こちらまで手を入れるかは未定。なのでbuildしたらupする必要はありません。