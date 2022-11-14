import os
from pathlib import Path

import music21
from music21 import corpus, converter

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

    for part in score.parts:
        cur_piece_str.append(TRACK_START)
        cur_track_str = preprocess_track(part, meta_info)
        cur_piece_str.extend(cur_track_str)
        cur_piece_str.append(TRACK_END)

    # print(track_list)
    cur_piece['MUSIC'] = track_list

    # cur_piece_str.append(PIECE_END)

    return cur_piece_str


def preprocess_track(track, meta_info):
    track_txt = [f'{INSTRUMENT}=0', 'DENSITY=1']
    # read current track
    for elem_part in track:
        if isinstance(elem_part, music21.instrument.Instrument):
            if str(elem_part) not in INSTR_DICT.keys():
                INSTR_DICT[str(elem_part)] = len(list(INSTR_DICT.keys()))
            # track_txt.append(f'{INSTRUMENT}={INSTR_DICT[str(elem_part)]}')
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

    prev_beat = 1.0
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
                bar_txt.append(f'{TIME_SHIFT}={elem_measure.duration.quarterLength * 4}')
            else:
                note_list = [f'{NOTE_ON}={elem_measure.pitch.midi}',
                             f'{TIME_SHIFT}={elem_measure.duration.quarterLength * 4}',
                             f'{NOTE_OFF}={elem_measure.pitch.midi}']

                cur_elem_duration = elem_measure.duration.quarterLength

                if elem_measure.beat - prev_duration > prev_beat:
                    shift_duration = elem_measure.beat - prev_duration - prev_beat

                    bar_txt.append(f'{TIME_SHIFT}='
                                   f'{shift_duration * 4}')

                    prev_duration += shift_duration

                bar_txt.extend(note_list)
                prev_duration += cur_elem_duration
                prev_beat += elem_measure.beat

        else:
            pass

    bar_dict['bar_txt'] = bar_txt
    return bar_dict


def extract_notes(file_list, parser, mode='build'):
    """
    Extract notes names and durations from score
    :param file_list:
    :param parser:
    :param mode:
    :return pieces: list with text representation of music
    """
    if mode == 'build':
        pieces_str = []

        for i, file in enumerate(file_list):
            logger.info(f'{i + 1} Parsing {file}')
            original_score = parser.parse(file)
            # original_score.show()
            try:
                cur_piece_list = preprocess_score(original_score)
                cur_piece_str = ' '.join(cur_piece_list)
                pieces_str.append(cur_piece_str)
            except ValueError as e:
                logger.warning(e)

        return pieces_str


def piece_to_str(piece):
    """
    Convert piece dict in string
    :param piece: piece
    :return: string representation of piece dict
    """
    piece_str = [PIECE_START]

    for music_elem in piece['MUSIC']:
        if music_elem == TRACK_START:
            piece_str.append(TRACK_START)
        else:
            for elem in music_elem:
                if INSTRUMENT in elem:
                    piece_str.append(elem)
                else:
                    if elem == BAR_START:
                        piece_str.append(BAR_START)
                    else:
                        for bar_elem in elem:
                            pass


def prepare_data(section, run_id, music_name):
    list_of_files, bach_parser = get_bach_chorales()

    store_folder, run_folder = create_dir_tree(section=section,
                                               run_id=run_id,
                                               music_name=music_name)

    piece_list_temp_repr = extract_notes(file_list=list_of_files[:5],
                                         parser=bach_parser,)

    print(piece_list_temp_repr)
    with open('text_repr.txt', 'w') as hfile:
        for piece in piece_list_temp_repr:
            hfile.write(piece + '\n')

    # for piece in piece_list_temp_repr:
    #     for elem in piece:
    #         if elem == PIECE_START \
    #                 or elem == TRACK_START \
    #                 or INSTRUMENT in elem \
    #                 or elem == BAR_START \
    #                 or elem == BAR_END \
    #                 or elem == TRACK_END \
    #                 or elem == PIECE_START:
    #             print()
    #             print(elem)
    #         else:
    #             print(elem, end='; ')

    # for track in piece_list_temp_repr[0]['MUSIC']:
    #     print(track)
    #
    # for piece in piece_list_temp_repr:
    #     piece_to_str(piece)


if __name__ == '__main__':
    # prepare_data(section='compose',
    #              run_id='0007',
    #              music_name='cello')

    text_repr = extract_notes(['data/v_lesu_elka.mid'], converter)

    with open('text_repr.txt', 'w') as hfile:
        for piece in text_repr:
            hfile.write(piece + '\n')
