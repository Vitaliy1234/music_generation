import pandas as pd

from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments


CLASSES = {'cheerful': 0,
           'tense': 1,
           'bizarre': 2}


def preprocess_midi_dataset(dataset_path, annotations_path):
    annots = pd.read_csv(annotations_path)
    annots = annots[annots['toptag_eng_verified'].isin(CLASSES.keys())]
    annots['toptag_eng_verified'] = annots['toptag_eng_verified'].replace(CLASSES).astype('float')

    args = MMMTokenizerArguments(
        prev_tokens="",
        windowing=True,
        time_unit="HUNDRED_TWENTY_EIGHT",
        num_programs=None,
        shuffle_tracks=True,
        track_density=False,
        window_size=32,
        hop_length=16,
        time_sig=True,
        velocity=True,
    )

