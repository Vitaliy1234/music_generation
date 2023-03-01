import os
from pathlib import Path
import random
import pandas as pd
from data_preparation import get_text_repr_filelist


def prepare_annotations(labels_file: str) -> None:
    """
    rename filenames in annotations from .wav to .mid
    :param labels_file:
    :return:
    """
    labels = pd.read_csv(labels_file)
    # filenames have .wav extension, but dataset consists of .mid
    labels['fname'] = labels['fname'].apply(lambda fname: fname.replace('.wav', '.mid'))
    labels.to_csv(labels_file, index=False)


def train_test_split_and_save(labels_file, class_labels):
    labels = pd.read_csv(labels_file)
    # choose classes in class_labels list
    labels = labels[labels['toptag_eng_verified'].isin(class_labels)]
    # split on train and test
    train = labels.sample(frac=0.8)
    test = labels[~labels.index.isin(train.index)]

    print(f'Train shape: {train.shape}, test shape: {test.shape}')

    train.reset_index(drop=True).to_csv('annotations_train.csv', index=False)
    test.reset_index(drop=True).to_csv('annotations_test.csv', index=False)


def build_structured_dataset(raw_dataset_path, annotations, output_dir, train_test_frac):
    """
    The function creates dir tree for dataset and store files in that tree in order to their classes
    :param train_test_frac: fraction of midi to use in test dataset
    :param raw_dataset_path: path to raw midi dataset
    :param annotations: file with emotion annotations
    :param output_dir: dir for text dataset
    :return:
    """
    # creating dirs for text-midi dataset with train-test division
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    Path(output_dir).mkdir(exist_ok=True)
    Path(train_dir).mkdir(exist_ok=True)
    Path(test_dir).mkdir(exist_ok=True)

    labels = pd.read_csv(annotations)

    # get text_repr of all midi files
    all_midi_files = []
    for file in os.listdir(raw_dataset_path):
        if file.endswith('.mid'):
            cur_midi_file = os.path.join(raw_dataset_path, file)
            all_midi_files.append(cur_midi_file)

    text_repr = get_text_repr_filelist(all_midi_files)
    # save text representations of midi in text files according to their classes
    for midi_file, text_midi in zip(all_midi_files, text_repr):
        cur_midi_file = os.path.split(midi_file)[1]
        cur_label = labels[labels['fname'] == cur_midi_file]['toptag_eng_verified'].item()

        # split text_midi to bars
        text_bars = []

        start_track = text_midi.index('TRACK_START') + len('TRACK_START') + 1
        end_track = text_midi.rfind('TRACK_END') - 1
        text_tracks = text_midi[start_track:end_track].split(' TRACK_END TRACK_START ')

        for text_track in text_tracks:
            start = text_track.index('BAR_START') + len('BAR_START') + 1
            end = text_track.rfind('BAR_END') - 1
            cur_text_bars = text_track[start:end].split(' BAR_END BAR_START ')

            # delete empty bars and one-note bars
            for text_bar in cur_text_bars:
                # we need at least two notes in bar
                if len(text_bar.split(' ')) >= 6:  # NOTE_ON TIME_DELTA NOTE_OFF NOTE_ON TIME_DELTA NOTE_OFF
                    text_bars.append(text_bar)

        if random.random() <= train_test_frac:
            cur_file_to_save = os.path.join(test_dir, cur_label) + '.txt'
        else:
            cur_file_to_save = os.path.join(train_dir, cur_label) + '.txt'

        with open(cur_file_to_save, 'a') as text_midi_file:
            text_midi_file.write('\n'.join(text_bars))


if __name__ == '__main__':
    labels_filename = 'verified_annotation.csv'
    dataset_path = 'emotion_midi'
    output_directory = 'emotion_midi_text'
    # prepare_annotations(labels_file=labels_filename)
    # classes = ['cheerful', 'tense']
    # factor = 0.2
    # train_test_split_and_save(labels_filename, classes)
    build_structured_dataset(dataset_path, labels_filename, output_directory, train_test_frac=0.3)

