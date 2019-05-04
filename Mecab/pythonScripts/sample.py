import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence

from os.path import isfile
import argparse

def get_all_files(directory):
    """
    指定されたディレクトリ配下の全ファイルのパスを再帰的に取得する
    ジェネレーター

    Parameters
    ----------
    directory : str
        再帰的に検索したいルートディレクトリ

    Returns
    -------
    generator : generator
        ディレクトリのパスを返すジェネレーター
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def read_document(path):
    """
    指定されたファイルをreadする

    Parameters
    ----------
    path : str
        readするファイルのフルパス

    Returns
    -------
    read : str
        ファイルをreadした結果str
    """
    with open(path, 'r', encoding='sjis', errors='ignore') as f:
        return f.read()

def trim_doc(doc):
    lines = doc.splitlines()
    valid_lines = []
    is_valid = False
    horizontal_rule_cnt = 0
    break_cnt = 0
    for line in lines:
        if horizontal_rule_cnt < 2 and '-----' in line:
            horizontal_rule_cnt += 1
            is_valid = horizontal_rule_cnt == 2
            continue
        if not(is_valid):
            continue
        if line == '':
            break_cnt += 1
            is_valid = break_cnt != 3
            continue
        break_cnt = 0
        valid_lines.append(line)
    return ''.join(valid_lines)

def split_into_words(doc, name=''):
    """
    文章を単語に分かち書きした結果を返却する

    Parameters
    ----------
    doc : str
        分かち書きする前の文章
    name : str
        文章の名称。任意

    Returns
    -------
    read : LabeledSentence
        分かち書きした結果
    """
    mecab = MeCab.Tagger("-Ochasen")
    valid_doc = trim_doc(doc)
    lines = mecab.parse(doc).splitlines()
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])

def corpus_to_sentences(corpus):
    """
    文章ファイルリストから分かち書きした結果を返す
    ジェネレーター

    Parameters
    ----------
    corpus : list
        分かち書きする前の文章のフルパスとラベルが格納されたlist

    Returns
    -------
    generator : generator
        分かち書きした結果を返すジェネレーター
    """
    docs = [read_document(x) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        sys.stdout.write('\r前処理中 {} / {}'.format(idx, len(corpus)))
        yield split_into_words(doc, name)

def train(sentences):

    def judge_early_stopping(model, sentences):
        ranks = []
        for doc_id in range(100):
            inferred_vector = model.infer_vector(sentences[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
            rank = [docid for docid, sim in sims].index(sentences[doc_id].tags[0])
            ranks.append(rank)
        print(collections.Counter(ranks))
        if collections.Counter(ranks)[0] >= PASSING_PRECISION:
            return True
        return False
    
    model = models.Doc2Vec(size=VEC_SIZE, alpha=TRAIN_ALPHA, sample=1e-4, min_count=1, workers=4)
    model.build_vocab(sentences)
    first_train_epochs = TRAIN_EPOCHS*4//5
    second_train_epochs = TRAIN_EPOCHS//5
    for x in range(first_train_epochs):
        print(str(x+1) + '/' + str(first_train_epochs))
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
        is_stopping = judge_early_stopping(model, sentences)
        if is_stopping:
            return model
    print('set alpha = alpha*0.8 for last 20% epochs')
    model.alpha = model.alpha * 0.8
    for x in range(second_train_epochs):
        print(str(x) + '/' + str(second_train_epochs))
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
        is_stopping = judge_early_stopping(model, sentences)
        if is_stopping:
            break
    return model
    
def tokenize(text):
    """
    読み込んだ文書ファイルから分かち書きした結果を返す

    Parameters
    ----------
    text : str
        分かち書きする前の文章

    Returns
    -------
    str : str
        分かち書きした結果
    """
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split()

def parse_args():
    parser = argparse.ArgumentParser(description=
                                     """
                                     Doc2Vecのサンプルプログラム。JPOPの歌詞で学習させたが著作権的に
                                     歌詞ファイルが提供できないので、配布用にlivedoorニュースのクリエイティブ・コモンズ
                                     ライセンスが適用されるニュース記事も入れている
                                     """)
    # モデル構築用設定値
    parser.add_argument('--docs'  , default='text/', type=str,
                        help='ドキュメントのルートディレクトリ。この配下の全txtファイルを学習する')
    parser.add_argument('--model' , default='model/news2vec.model', type=str,
                        help='学習済みモデルのパス。あれば読み込み無ければ学習後に保存する')
    # 学習用設定値
    parser.add_argument('--epochs', default=50, type=int, help='学習するEpoch数')
    parser.add_argument('--alpha' , help='学習率のようなもの', default=0.015, type=float)
    parser.add_argument('--vsize' , help='ベクトルの次元', default=400, type=int)
    # 推論用設定値
    parser.add_argument('--depth' , default=None, type=int,
                        help='作者毎にディレクトリを切っていた場合設定する。lyrics/artistID/曲.txt の場合は2')
    parser.add_argument('--pred_f', default='./text/livedoor-homme/livedoor-homme-5625149.txt', type=str,
                        help='類似度を確認したいファイルのフルパス。')

    args = parser.parse_args()
    return args
    

if __name__ == '__main__':

    args = parse_args()
    PRE_TRAIN_MODEL_PATH = args.model
    MODEL_DIR            = PRE_TRAIN_MODEL_PATH.split('/')[-2]
    SAMPLE_TEXT_ROOT_DIR = args.docs
    TRAIN_EPOCHS         = int(args.epochs)
    TRAIN_ALPHA          = float(args.alpha)
    ARTIST_ID_DEPTH      = args.depth
    VEC_SIZE             = int(args.vsize)
    PREDICT_FILE         = args.pred_f
    PASSING_PRECISION    = 93

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    # 訓練済みモデルがあればそのまま推論に進み、なければコーパス作成から学習までを実施
    if isfile(PRE_TRAIN_MODEL_PATH):
        print('訓練済みモデルを使用します')
        model = models.Doc2Vec.load(PRE_TRAIN_MODEL_PATH)
    else:
        corpus = list(get_all_files(SAMPLE_TEXT_ROOT_DIR))
        sentences = list(corpus_to_sentences(corpus))
        print()
        model = train(sentences)
        model.save(PRE_TRAIN_MODEL_PATH)
    
    print('類似度検索対象 : ' + PREDICT_FILE)
    predict_text = read_document(PREDICT_FILE)
    predict_results = model.docvecs.most_similar([model.infer_vector(tokenize(predict_text))], topn=10)
    if ARTIST_ID_DEPTH is not None:
        for result in predict_results:
            print('分類 : ' + result[0].split('/')[int(ARTIST_ID_DEPTH) - 1], 
                  '類似度 : ' + str(result[1]), 
                  'ファイルパス : ' + result[0]
                 )
    else:
        for result in predict_results:
            print('ファイルパス : ' + result[0], 
                  '類似度 : ' + str(result[1])
                 )

