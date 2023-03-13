import os

import re

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import gensim

from emotion_classification.preprocess_miditok import miditok_test

from data_preparation import transpose_text_midi


def word_vector(model_w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


def start(dataset, annotations):
    annots = pd.read_csv(annotations)
    annots = annots[annots['toptag_eng_verified'].isin(['cheerful', 'tense'])]

    classes = {'cheerful': 0,
               'tense': 1}

    annots['toptag_eng_verified'] = annots['toptag_eng_verified'].replace(classes).astype('float')

    midi_texts = []

    for txt_midi in tqdm(annots['fname']):
        # read encoded midi via MMM Encoding in my realization
        with open(os.path.join(dataset, txt_midi.replace('.mid', '.txt')), 'r') as t_mid:
            midi_texts.append(t_mid.read())

    annots['midi_text'] = midi_texts

    split_index = int(annots.shape[0] * 0.7)
    annots_train = annots[:split_index]
    annots_test = annots[split_index:]

    print(annots_train['toptag_eng_verified'].value_counts())
    print(annots_test['toptag_eng_verified'].value_counts())

    annots_train['midi_text'] = annots_train['midi_text'].apply(lambda elem: transpose_text_midi(elem, range(12)))
    annots_train = annots_train.explode('midi_text')

    # vectorizer = TfidfVectorizer(token_pattern=r'bar_start.+?bar_end', min_df=3, max_df=0.7, ngram_range=(1, 2))

    # gensim
    vec_size = 200
    tokenizer = re.compile(r'BAR_START.+?BAR_END')
    tokenized_midi = list(annots_train['midi_text'].apply(lambda text: tokenizer.findall(text)))
    music_sentences = tokenized_midi
    music_sentences.extend(list(annots_test['midi_text'].apply(lambda text: tokenizer.findall(text))))

    vectorizer = gensim.models.Word2Vec(sentences=music_sentences, vector_size=vec_size, window=15, min_count=5)
    bar_vectors = vectorizer.wv

    word2vec_arrays = np.zeros((len(music_sentences), vec_size))
    for i in range(len(music_sentences)):
        word2vec_arrays[i, :] = word_vector(bar_vectors, music_sentences[i], vec_size)
    word2vec_df = pd.DataFrame(word2vec_arrays)

    # X_train = annots_train['midi_text']
    # y_train = annots_train['toptag_eng_verified']
    # X_test = annots_test['midi_text']
    # y_test = annots_test['toptag_eng_verified']
    #
    # X_train = vectorizer.fit_transform(X_train)
    # X_test = vectorizer.transform(X_test)

    X_train = word2vec_df.iloc[:annots_train.shape[0]]
    y_train = annots_train['toptag_eng_verified']
    X_test = word2vec_df.iloc[annots_train.shape[0]:]
    y_test = annots_test['toptag_eng_verified']

    print(X_train.shape)
    print(X_test.shape)

    model = SVC(random_state=0, tol=1e-4)
    model.fit(X_train, y_train)

    ytest = np.array(y_test)

    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))
    print(model.coef_)

    bar_to_coef = {}

    # for bar in vectorizer.vocabulary_:
    #     bar_to_coef[bar] = model.coef_[0][vectorizer.vocabulary_[bar]]
    #
    # bar_weights = pd.DataFrame(bar_to_coef.values(), index=list(bar_to_coef.keys()), columns=['weight'])
    # bar_weights.to_excel('bar_weights.xlsx')
    return


if __name__ == '__main__':
    # dataset_path = '/Users/18629082/Desktop/music_generation/data/music_midi/emotion_midi_texts'
    # annotation_path = '/Users/18629082/Desktop/music_generation/data/music_midi/verified_annotation.csv'
    dataset_path = r'D:\Диссер_музыка\music_generation\data\music_midi\emotion_midi_texts'
    annotation_path = r'D:\Диссер_музыка\music_generation\data\music_midi\verified_annotation.csv'
    start(dataset_path, annotation_path)
