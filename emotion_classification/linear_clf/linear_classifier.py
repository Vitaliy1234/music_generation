import os

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from emotion_classification.preprocess_miditok import miditok_test

from data_preparation import transpose_text_midi


def start(dataset, annotations):
    annots = pd.read_csv(annotations)
    annots = annots[annots['toptag_eng_verified'].isin(['cheerful', 'tense'])]

    classes = {'cheerful': 0,
               'tense': 1,}

    annots['toptag_eng_verified'] = annots['toptag_eng_verified'].replace(classes).astype('float')

    midi_texts = []

    for txt_midi in tqdm(annots['fname']):
        # Encode midi via miditok library
        # cur_midi_file = os.path.join(dataset, txt_midi)
        # text = miditok_test(cur_midi_file)
        #
        # midi_texts.append(text)

        # encode midi via MMM Encoding in my realization
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

    vectorizer = TfidfVectorizer(token_pattern=r'bar_start.+?bar_end', min_df=5, max_df=0.7, ngram_range=(1, 2))
    # vectorizer = TfidfVectorizer(token_pattern=r'ar_none.+?b', min_df=1, max_df=0.1, ngram_range=(1, 1))
    # vectorizer = TfidfVectorizer(token_pattern=r' .+? ', ngram_range=(1, 7), min_df=3, max_df=0.8)

    X_train = annots_train['midi_text']
    y_train = annots_train['toptag_eng_verified']
    X_test = annots_test['midi_text']
    y_test = annots_test['toptag_eng_verified']

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print(X_train.shape)
    print(X_test.shape)
    print(vectorizer.get_feature_names_out())

    model = SVC(random_state=0)
    model.fit(X_train, y_train)

    ytest = np.array(y_test)

    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))
    print(model.coef_)

    bar_to_coef = {}

    for bar in vectorizer.vocabulary_:
        bar_to_coef[bar] = model.coef_[0][vectorizer.vocabulary_[bar]]

    bar_weights = pd.DataFrame(bar_to_coef.values(), index=list(bar_to_coef.keys()), columns=['weight'])
    bar_weights.to_excel('bar_weights.xlsx')
    return


if __name__ == '__main__':
    dataset_path = '/Users/18629082/Desktop/music_generation/data/music_midi/emotion_midi_texts'
    annotation_path = '/Users/18629082/Desktop/music_generation/data/music_midi/verified_annotation.csv'
    start(dataset_path, annotation_path)
