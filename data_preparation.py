import os
from pathlib import Path
import numpy as np

import music21
from music21 import corpus, converter, interval, pitch

from helpers import logging

logger = logging.create_logger("data_preparation")


BAR_START = 'BAR_START'
BAR_END = 'BAR_END'
TRACK_START = 'TRACK_START'
TRACK_END = 'TRACK_END'
NOTE_ON = 'NOTE_ON'
NOTE_OFF = 'NOTE_OFF'
TIME_SHIFT = 'TIME_DELTA'
INSTRUMENT = 'INST'

INSTR_DICT = {}

PIECE_START = 'PIECE_START'
PIECE_END = 'PIECE_END'


def get_bach_chorales():
    """
    Read Bach's chorales
    :return file_list, parser: list of midi files and parser to read them
    """
    file_list = ['bwv' + str(x['bwv']) for x in corpus.chorales.ChoraleList().byBWV.values()]
    parser = corpus

    return file_list, parser


def create_dir_tree(section, run_id, music_name):
    """
    Function for creating directory tree to work with dataset
    :param section:
    :param run_id:
    :param music_name:
    :return store_folder: where to store dataset in an appropriate state
    """
    run_folder = os.path.join('run', section)
    run_folder = os.path.join(run_folder, '_'.join([run_id, music_name]))
    store_folder = os.path.join(run_folder, 'store')

    Path(store_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'output')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'weights')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'viz')).mkdir(parents=True, exist_ok=True)

    return store_folder, run_folder


def preprocess_score(score):
    cur_piece_str = [PIECE_START]
    cur_piece = {}
    meta_info = {}
    track_list = []

    for part_index, part in enumerate(score.parts):
        cur_piece_str.append(TRACK_START)
        cur_track_str = preprocess_track(part, part_index, meta_info)
        cur_piece_str.extend(cur_track_str)
        cur_piece_str.append(TRACK_END)

    # print(track_list)
    cur_piece['MUSIC'] = track_list

    # cur_piece_str.append(PIECE_END)

    return cur_piece_str


def preprocess_track(track, track_index, meta_info):
    track_txt = [f'{INSTRUMENT}={track_index}', 'DENSITY=1']
    # read current track
    for elem_part in track:
        if isinstance(elem_part, music21.instrument.Instrument):
            if str(elem_part) not in INSTR_DICT.keys():
                INSTR_DICT[str(elem_part)] = len(list(INSTR_DICT.keys()))
            # track_txt.append(f'{INSTRUMENT}={INSTR_DICT[str(elem_part)]}')
            # track_txt.append('DENSITY=1')
        elif isinstance(elem_part, music21.stream.base.Measure):
            track_txt.append(BAR_START)
            cur_bar_info = preprocess_bar(elem_part)

            for info_key in ['Key', 'Beat duration', 'Beat count']:
                if info_key in cur_bar_info.keys() and info_key not in meta_info.keys():
                    meta_info[info_key] = cur_bar_info[info_key]
                elif info_key in cur_bar_info.keys() and info_key in meta_info.keys():
                    if cur_bar_info[info_key] != meta_info[info_key]:
                        raise ValueError('Key or time signature was changed')

            cur_bar_time_sig = meta_info['Beat count']
            # case when there is empty bar
            if not cur_bar_info['bar_txt']:
                track_txt.append(f'{TIME_SHIFT}={cur_bar_time_sig * 4}')
            else:
                track_txt.extend(cur_bar_info['bar_txt'])

            track_txt.append(BAR_END)

        else:
            pass

    return track_txt


def preprocess_bar(bar):
    bar_txt = []
    bar_dict = {}

    prev_offset = 0.0
    prev_duration = 0.0
    # read measure
    for elem_measure in bar:
        if isinstance(elem_measure, music21.key.Key):
            bar_dict['Key'] = str(elem_measure.asKey())
        elif isinstance(elem_measure, music21.meter.base.TimeSignature):
            bar_dict['Beat duration'] = str(elem_measure.beatDuration.quarterLength)
            bar_dict['Beat count'] = elem_measure.beatCount
            bar_dict['Time signature'] = elem_measure

        elif isinstance(elem_measure, music21.note.Note):
            if elem_measure.isRest:
                bar_txt.append(f'{TIME_SHIFT}={float(elem_measure.duration.quarterLength) * 16}')
            else:
                note_list = [f'{NOTE_ON}={elem_measure.pitch.midi}',
                             f'{TIME_SHIFT}={float(elem_measure.duration.quarterLength) * 16}',
                             f'{NOTE_OFF}={elem_measure.pitch.midi}']

                cur_elem_duration = elem_measure.duration.quarterLength

                if elem_measure.offset - prev_offset > prev_duration:
                    shift_duration = elem_measure.offset - prev_duration - prev_offset

                    if bar_txt and False:
                        note_off_token = bar_txt.pop()
                        prev_time_shift = bar_txt.pop()

                        bar_txt.append(f'{TIME_SHIFT}='
                                       f'{shift_duration * 16 + float(prev_time_shift.split("=")[1])}')
                        bar_txt.append(note_off_token)
                    else:
                        bar_txt.append(f'{TIME_SHIFT}='
                                       f'{shift_duration * 16}')

                    prev_duration += shift_duration

                bar_txt.extend(note_list)
                prev_duration += cur_elem_duration
                prev_offset = elem_measure.offset

        else:
            pass

    bar_dict['bar_txt'] = bar_txt
    return bar_dict


def extract_notes(file_list, mode='build'):
    """
    Extract notes names and durations from score
    :param file_list:
    :param mode:
    :return pieces: list with text representation of music
    """
    if mode == 'build':
        pieces_str = []

        for i, file in enumerate(file_list):
            logger.info(f'{i + 1} Parsing {file}')
            original_score = converter.parse(file)

            # original_score.show()
            try:
                cur_piece_list = preprocess_score(original_score)
                cur_piece_str = ' '.join(cur_piece_list)
                pieces_str.append(cur_piece_str)
            except ValueError as e:
                logger.warning(e)

        return pieces_str


def midi_to_text(parser, file):
    logger.info(f'Parsing {file}')
    original_score = parser.parse(file)

    try:
        cur_piece_list = preprocess_score(original_score)
        cur_piece_str = ' '.join(cur_piece_list)
        return cur_piece_str
    except ValueError as e:
        logger.warning(e)


def prepare_data(section, run_id, music_name):
    list_of_files, bach_parser = get_bach_chorales()

    store_folder, run_folder = create_dir_tree(section=section,
                                               run_id=run_id,
                                               music_name=music_name)

    piece_list_temp_repr = extract_notes(file_list=list_of_files[:5],)

    print(piece_list_temp_repr)
    with open('text_repr.txt', 'w') as hfile:
        for piece in piece_list_temp_repr:
            hfile.write(piece + '\n')


def get_text_repr_file(midi_file):
    """
    Makes text representation from midi file
    :param midi_file:
    :return: text representation
    """
    text_representation = midi_to_text(converter, midi_file)

    return text_representation


def get_text_repr_filelist(file_list):
    text_representation = extract_notes(file_list=file_list)
    return text_representation


def transpose_text_midi(text_midi, transpositions):
    result = {}

    for transposition in transpositions:
        result[transposition] = []
        for token in text_midi.split(' '):
            if 'NOTE_ON' in token or 'NOTE_OFF' in token:
                cur_pitch_value = int(token.split('=')[1])
                cur_token_without_value = token.split('=')[0]
                result[transposition].append(f'{cur_token_without_value}={cur_pitch_value + transposition}')
            else:
                result[transposition].append(token)

        result[transposition] = ' '.join(result[transposition])

    return np.array(list(result.values()))


def to_interval_repr(text_midi):
    pass


if __name__ == '__main__':
    # prepare_data(section='compose',
    #              run_id='0007',
    #              music_name='cello')

    text_repr = extract_notes(['data/99_basic_pitch.mid'])

    with open('text_repr.txt', 'w') as hfile:
        for piece in text_repr:
            hfile.write(piece + '\n')
