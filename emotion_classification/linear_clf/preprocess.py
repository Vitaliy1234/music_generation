from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments
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


if __name__ == '__main__':
    file = ['/Users/18629082/Desktop/music_generation/data/test.mid']
    preprocess_midi(file)
