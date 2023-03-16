import os
from pathlib import Path

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from miditok import CPWord, REMI


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


# for miditok tokenizer
def midi_valid(midi) -> bool:
    if any(ts.numerator != 4 for ts in midi.time_signature_changes):
        return False  # time signature different from 4/*, 4 beats per bar
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True


def start(dataset, annotations):
    classes = {'cheerful': 0,
               'tense': 1}

    annots = pd.read_csv(annotations)
    annots = annots[annots['toptag_eng_verified'].isin(classes.keys())]

    annots['toptag_eng_verified'] = annots['toptag_eng_verified'].replace(classes).astype('float')

    midi_texts = []
    # experiment with miditok
    tokens_nobpe_path = Path('tokenized_dataset', 'tokens_noBPE')
    tokens_nobpe_path.mkdir(exist_ok=True, parents=True)
    tokens_bpe_path = Path('tokenized_dataset', 'tokens_BPE')
    tokens_bpe_path.mkdir(exist_ok=True)
    tokenizer = REMI()
    data_augmentation_offsets = [2, 2, 2]

    midi_dataset = list(annots['fname'].apply(lambda fname: os.path.join(dataset, fname)).values)
    tokenizer.tokenize_midi_dataset(midi_dataset,
                                    tokens_nobpe_path,
                                    midi_valid,
                                    data_augmentation_offsets)
    # learn BPE
    # tokenizer.learn_bpe(
    #     vocab_size=500,
    #     tokens_paths=list(tokens_nobpe_path.glob("**/*.json")),
    # )
    #
    # tokenizer.apply_bpe_to_dataset(tokens_nobpe_path, tokens_bpe_path)

    annots['fname_short'] = annots['fname'].apply(lambda fname: fname.split('.mid')[0])
    annots['midi_text'] = [[]] * annots.shape[0]

    reversed_vocab = {ind: token for token, ind in zip(tokenizer.vocab.keys(), tokenizer.vocab.values())}

    X = []
    y = []
    print('Creating dataset for train and test model')
    for file in tokens_nobpe_path.glob("**/*.json"):
        tokens = tokenizer.load_tokens(file)
        ids_list = tokens['ids']
        str_tokens = []

        for track in ids_list:
            track_str_tokens = [reversed_vocab[cur_id] for cur_id in track]

            str_tokens.extend(track_str_tokens)

        midi_texts.append(' '.join(str_tokens))
        # name of current file without extension and § (used in miditok augmentation)
        file_short = file.parts[-1].split('.json')[0].split('§')[0]
        label = annots[annots['fname_short'] == file_short]['toptag_eng_verified'].values
        label = label[0]

        X.append(' '.join(str_tokens))
        y.append(label)

    vectorizer = TfidfVectorizer(token_pattern=r'ar_none.+?b', min_df=3, max_df=0.8, ngram_range=(1, 1))

    X = pd.DataFrame(X, columns=['midi_text'])
    y = np.array(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    X_train = vectorizer.fit_transform(X_train['midi_text'])
    X_test = vectorizer.transform(X_test['midi_text'])

    print(X_train.shape)
    print(X_test.shape)

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    ytest = np.array(y_test)

    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))
    print(model.coef_)

    # save bar coefficients in excel file
    bar_to_coef = {}
    for bar in vectorizer.vocabulary_:
        bar_to_coef[bar] = model.coef_[0][vectorizer.vocabulary_[bar]]

    bar_weights = pd.DataFrame(bar_to_coef.values(), index=list(bar_to_coef.keys()), columns=['weight'])
    bar_weights.to_excel('bar_weights.xlsx')
    return


if __name__ == '__main__':
    # dataset_path = '/Users/18629082/Desktop/music_generation/data/music_midi/emotion_midi_texts'
    dataset_path = '/Users/18629082/Desktop/music_generation/data/music_midi/emotion_midi'
    annotation_path = '/Users/18629082/Desktop/music_generation/data/music_midi/verified_annotation.csv'
    # dataset_path = r'D:\Диссер_музыка\music_generation\data\music_midi\emotion_midi_texts'
    # annotation_path = r'D:\Диссер_музыка\music_generation\data\music_midi\verified_annotation.csv'
    start(dataset_path, annotation_path)
