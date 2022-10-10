import os
from pathlib import Path

import pickle

import music21
from music21 import corpus, note, chord


def prepare_data(section, run_id, music_name):
    list_of_files, bach_parser = get_bach_chorales()

    store_folder, run_folder = create_dir_tree(section=section,
                                               run_id=run_id,
                                               music_name=music_name)

    notes, durations = extract_notes(file_list=list_of_files,
                                     parser=bach_parser,
                                     store_folder=store_folder)
    print(notes)
    print(durations)


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


def extract_notes(file_list, parser, store_folder, mode='build'):
    """
    Extract notes names and durations from score
    :param file_list:
    :param parser:
    :param store_folder:
    :param mode:
    :return notes, durations: list of notes names and list of durations
    """
    if mode == 'build':

        for i, file in enumerate(file_list):
            print(i + 1, "Parsing %s" % file)
            original_score = parser.parse(file)
            original_score.show()

            for part in original_score.parts:
                track_txt = []
                # read current part
                for elem_part in part:
                    if isinstance(elem_part, music21.instrument.Instrument):
                        # print(f'Instrument: {elem_part}')
                        track_txt.append(f'INST={elem_part}')
                        pass
                    elif isinstance(elem_part, music21.stream.base.Measure):
                        # print(f'Measure: {elem_part}')
                        track_txt.append('BAR_START')

                        bar_txt = []
                        # read measure
                        for elem_measure in elem_part:
                            if isinstance(elem_measure, music21.key.Key):
                                # print(f'Key: {elem_measure}')
                                pass
                            elif isinstance(elem_measure, music21.meter.base.TimeSignature):
                                print(f'Beat duration: {elem_measure.beatDuration}')
                                print(f'Beat count: {elem_measure.beatCount}')
                                print(f'Time signature: {elem_measure}')
                                pass
                            elif isinstance(elem_measure, music21.note.Note):
                                if elem_measure.isRest:
                                    print('rest')
                                    bar_txt.append(f'TIME_SHIFT={elem_measure.duration.quarterLength}')
                                else:
                                    # print(f'duration: {elem_measure.duration}')
                                    # print(f'Name: {elem_measure.nameWithOctave}')

                                    bar_txt.append(f'NOTE_ON={elem_measure.nameWithOctave}')
                                    bar_txt.append(f'TIME_SHIFT={elem_measure.duration.quarterLength}')
                                    bar_txt.append(f'NOTE_OFF={elem_measure.nameWithOctave}')

                            else:
                                # print(type(elem_measure))
                                pass
                        track_txt.append(bar_txt)
                        track_txt.append('BAR_END')
                    else:
                        # print(type(elem_part))
                        pass

                print(track_txt)

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

            return None, None


if __name__ == '__main__':
    prepare_data(section='compose',
                 run_id='0007',
                 music_name='cello')
