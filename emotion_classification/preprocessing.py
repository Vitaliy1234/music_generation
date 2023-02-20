import os
from pathlib import Path

import pandas as pd
from data_preparation import get_text_repr


def preprocess(input_dir: str, output_file: str) -> None:
    """

    :param input_dir:
    :param output_file:
    :return:
    """
    files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)[:3]]
    texts = get_text_repr(files=files)

    with open(output_file, 'w') as txt_music:
        txt_music.writelines(texts)


if __name__ == '__main__':
    dataset = '../data/music_midi/emotion_midi'
    output = '../data/music_midi/emotion_midi_text/'
    Path(output).mkdir(exist_ok=True)
    output_file = os.path.join(output, 'dataset.txt')
    preprocess(input_dir=dataset, output_file=output_file)
