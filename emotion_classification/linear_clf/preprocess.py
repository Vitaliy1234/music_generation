import os
from pathlib import Path

from data_preparation import get_text_repr_filelist

from bar_tokenizer import BarTokenizer


def preprocess_midi(midi_file_list):
    midi_texts = get_text_repr_filelist(midi_file_list)
    result = []

    for text_midi in midi_texts:
        bar_tok = BarTokenizer(text_midi)
        text_bars = bar_tok.tokenize_text_midi()

        result.append(' '.join(text_bars))

    return result


def midi2textfiles(midi_file_list):
    midi_texts = get_text_repr_filelist(midi_file_list)

    print(len(midi_texts), len(midi_file_list))

    for text_midi, midi_filename in zip(midi_texts, midi_file_list):
        print(midi_filename)
        cur_midi_file = os.path.split(midi_filename)[1]
        with open(f'../../data/music_midi/emotion_midi_texts_new/{cur_midi_file.replace(".mid", ".txt")}', 'w') as hfile:
            hfile.write(text_midi)


if __name__ == '__main__':
    files = Path('../../data/music_midi/emotion_midi').glob('*.mid')
    midi2textfiles(list(files))
