import os
from pathlib import Path

import music21
from music21 import corpus, note, chord


BAR_START = '<BAR_START>'
BAR_END = '<BAR_END>'
TRACK_START = '<TRACK_START>'
TRACK_END = '<TRACK_END>'
NOTE_ON = '<NOTE_ON>'
NOTE_OFF = '<NOTE_OFF>'
TIME_SHIFT = '<TIME_SHIFT>'
INSTRUMENT = '<INST>'

PIECE_START = '<PIECE_START>'
PIECE_END = '<PIECE_END>'


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


def extract_notes(file_list, parser, mode='build'):
    """
    Extract notes names and durations from score
    :param file_list:
    :param parser:
    :param mode:
    :return pieces: list with text representation of music
    """
    if mode == 'build':
        pieces = []

        pieces_str = []

        for i, file in enumerate(file_list):
            print(i + 1, "Parsing %s" % file)
            original_score = parser.parse(file)
            # original_score.show()
            cur_piece = {}
            cur_piece_str = [PIECE_START]
            piece_key = ''
            piece_time_signature = None
            track_list = []

            for part in original_score.parts:
                cur_piece_str.append(TRACK_START)
                track_txt = []
                # read current part
                for elem_part in part:
                    if isinstance(elem_part, music21.instrument.Instrument):
                        # print(f'Instrument: {elem_part}')
                        track_txt.append(f'{INSTRUMENT}={elem_part}')
                        cur_piece_str.append(f'{INSTRUMENT}={elem_part}')
                        pass
                    elif isinstance(elem_part, music21.stream.base.Measure):
                        # print(f'Measure: {elem_part}')
                        # print(f'Measure num={elem_part.measureNumber}')

                        track_txt.append(BAR_START)
                        cur_piece_str.append(BAR_START)

                        bar_txt = []
                        bar_dict = {}

                        prev_beat = 1.0
                        prev_duration = 0.0
                        # read measure
                        for elem_measure in elem_part:
                            if isinstance(elem_measure, music21.key.Key):
                                # print(f'Key: {elem_measure.asKey()}')
                                cur_piece['Key'] = str(elem_measure.asKey())
                                piece_key = str(elem_measure.asKey())
                            elif isinstance(elem_measure, music21.meter.base.TimeSignature):
                                # print(f'Beat duration: {elem_measure.beatDuration}')
                                # print(f'Beat count: {elem_measure.beatCount}')
                                # print(f'Time signature: {elem_measure}')
                                cur_piece['Beat duration'] = str(elem_measure.beatDuration.quarterLength)
                                cur_piece['Beat count'] = elem_measure.beatCount
                                cur_piece['Time signature'] = elem_measure
                                piece_time_signature = elem_measure

                            elif isinstance(elem_measure, music21.note.Note):
                                if elem_measure.isRest:
                                    # print('rest')
                                    bar_txt.append(f'{TIME_SHIFT}={elem_measure.duration.quarterLength}')
                                    cur_piece_str.append(f'{TIME_SHIFT}={elem_measure.duration.quarterLength}')
                                else:
                                    # print(f'duration: {elem_measure.duration}')
                                    # print(f'Name: {elem_measure.nameWithOctave}')
                                    note_list = [f'{NOTE_ON}={elem_measure.nameWithOctave}',
                                                 f'{TIME_SHIFT}={elem_measure.duration.quarterLength}',
                                                 f'{NOTE_OFF}={elem_measure.nameWithOctave}']

                                    cur_elem_duration = elem_measure.duration.quarterLength

                                    if elem_measure.beat - prev_duration > prev_beat:
                                        shift_duration = elem_measure.beat - prev_duration - prev_beat
                                        cur_piece_str.append(f'{TIME_SHIFT}='
                                                             f'{shift_duration}')

                                        prev_duration += shift_duration

                                    cur_piece_str.extend(note_list)

                                    prev_duration += cur_elem_duration
                                    prev_beat += elem_measure.beat

                                    bar_txt.append({elem_measure.beat: note_list})

                            else:
                                # print(type(elem_measure))
                                pass
                        bar_dict[elem_part.measureNumber] = bar_txt
                        track_txt.append(bar_dict)
                        track_txt.append(BAR_END)

                        # case when there is empty bar
                        if cur_piece_str[-1] == BAR_START:
                            cur_piece_str.append(f'{TIME_SHIFT}={piece_time_signature.beatCount}')
                        cur_piece_str.append(BAR_END)
                    else:
                        # print(type(elem_part))
                        pass
                track_list.append(TRACK_START)
                track_list.append(track_txt)
                track_list.append(TRACK_END)
                cur_piece_str.append(TRACK_END)

            # print(track_list)
            cur_piece['MUSIC'] = track_list
            pieces.append(cur_piece)

            cur_piece_str.append(PIECE_END)

            pieces_str.append(cur_piece_str)

        # print(pieces[0])
        # save notes and durations
    #     with open(os.path.join(store_folder, 'notes'), 'wb') as notes_file:
    #         pickle.dump(notes, notes_file)
    #     with open(os.path.join(store_folder, 'durations'), 'wb') as durations_file:
    #         pickle.dump(durations, durations_file)
    # else:
    #     with open(os.path.join(store_folder, 'notes'), 'rb') as notes_file:
    #         notes = pickle.load(notes_file)
    #     with open(os.path.join(store_folder, 'durations'), 'rb') as durations_file:
    #         durations = pickle.load(durations_file)

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

    piece_list_temp_repr = extract_notes(file_list=list_of_files[:1],
                                         parser=bach_parser,)

    for piece in piece_list_temp_repr:
        for elem in piece:
            if elem == PIECE_START \
                    or elem == TRACK_START \
                    or INSTRUMENT in elem \
                    or elem == BAR_START \
                    or elem == BAR_END \
                    or elem == TRACK_END \
                    or elem == PIECE_START:
                print()
                print(elem)
            else:
                print(elem, end='; ')

    # for track in piece_list_temp_repr[0]['MUSIC']:
    #     print(track)
    #
    # for piece in piece_list_temp_repr:
    #     piece_to_str(piece)


if __name__ == '__main__':
    prepare_data(section='compose',
                 run_id='0007',
                 music_name='cello')
