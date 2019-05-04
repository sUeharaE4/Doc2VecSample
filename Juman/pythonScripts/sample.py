import sys
import os
from os import listdir, path
from os.path import isfile
from pyknp import Jumanpp
from gensim import models
from gensim.models.doc2vec import LabeledSentence

MODEL_DIR = 'model/'
PRE_TRAIN_MODEL_PATH = 'model/doc2vec.model'
SAMPLE_TEXT_ROOT_DIR = 'text/'

def corpus_files():
    dirs = [path.join(SAMPLE_TEXT_ROOT_DIR, x)
            for x in listdir(SAMPLE_TEXT_ROOT_DIR) if not x.endswith('.txt')]
    docs = [path.join(x, y)
            for x in dirs for y in listdir(x) if not x.startswith('LICENSE')]
    return docs

def read_document(path):
    with open(path, 'r') as f:
        return f.read()

def split_into_words(text):
    result = Jumanpp().analysis(text)
    return [mrph.midasi for mrph in result.mrph_list()]

def doc_to_sentence(doc, name):
    words = split_into_words(doc)
    return LabeledSentence(words=words, tags=[name])

def corpus_to_sentences(corpus):
    docs   = [read_document(x) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        sys.stdout.write('\r前処理中 {}/{}'.format(idx, len(corpus)))
        yield doc_to_sentence(doc, name)

corpus = corpus_files()
sentences = corpus_to_sentences(corpus)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if isfile(PRE_TRAIN_MODEL_PATH):
    print('訓練済みモデルを使用します')
    model = models.Doc2Vec.load(PRE_TRAIN_MODEL_PATH)
else:
    model = models.Doc2Vec(dm=0, size=300, window=15, alpha=.025,
                           min_alpha=.025, min_count=1, sample=1e-6
                          )
    model.build_vocab(sentences)
    print('\n訓練開始')
    for epoch in range(20):
        print('Epoch: {}'.format(epoch + 1))
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
        model.alpha -= (0.025 - 0.0001) / 19
        model.min_alpha = model.alpha
    model.save(PRE_TRAIN_MODEL_PATH)

predict_file = './text/livedoor-homme/livedoor-homme-5625149.txt'
print('類似度検索対象 : ' + predict_file)
predict_text = read_document(predict_file)
predict_results = model.docvecs.most_similar([model.infer_vector(split_into_words(predict_text))], topn=5)
for result in predict_results:
    print('類似度 : ' + str(result[1]), 
          'ファイルパス : ' + result[0]
         )
