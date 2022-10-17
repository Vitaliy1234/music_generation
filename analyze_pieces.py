from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd

from music21 import note, chord

from data_preparation import get_bach_chorales, extract_notes, NOTE_ON, TIME_SHIFT


PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B-', 'B']


def get_pitch_name_without_octave(pitch):
    return ''.join(pitch.split('=')[1][:-1])


def pitch_count(sample):
    """
    Counts unique pitches with octave-dependency
    :param sample: music composition
    :return: number of pitches
    """
    pitches = set()

    for elem in sample:
        if NOTE_ON in elem:
            pitches.add(elem)

    return len(pitches)


def pitch_class_histogram(sample):
    """
    Plot pitch class histogram: an octave-independent representation of
    the pitch content with a dimensionality of 12 for a
    chromatic scale
    :param sample: music
    :return: nothing - saves plot in file
    """
    pitches = defaultdict(int)

    for elem in sample:
        if NOTE_ON in elem:
            cur_note_without_octave = get_pitch_name_without_octave(elem)
            pitches[cur_note_without_octave] += 1

    pitches_df = pd.DataFrame(pitches.values(), index=pitches.keys(), columns=['note_count'])
    pitches_df.sort_index(inplace=True)

    plt.style.use('seaborn')
    plt.figure(figsize=(6, 5))
    plt.title('Pitch class histogram')
    sns.barplot(x=pitches_df.index, y=pitches_df['note_count'])
    plt.xlabel('Pitch name')
    plt.ylabel('Pitches Count')
    plt.savefig('pitch_class_histogram.jpg')
    # plt.show()


def pitch_class_transition_matrix(sample):
    """
    Plot pitch class transition matrix: histogram-like representation computed by counting
    the pitch transitions for each (ordered) pair of notes.
    :param sample: music composition
    :return: nothing - saves plot in file
    """
    pitches_arr = np.zeros((len(PITCHES), len(PITCHES)))
    prev_pitch = ''

    for elem in sample:
        if NOTE_ON in elem:
            cur_pitch = get_pitch_name_without_octave(elem)

            if prev_pitch == '':
                prev_pitch = cur_pitch
                continue

            pitches_arr[PITCHES.index(prev_pitch), PITCHES.index(cur_pitch)] += 1
            prev_pitch = cur_pitch

    pitches_df = pd.DataFrame(pitches_arr, index=PITCHES, columns=PITCHES)
    plt.figure(figsize=(7, 6))
    plt.title('Pitch class transition matrix')
    sns.heatmap(pitches_df, color='blue')
    plt.yticks(rotation=0)
    plt.savefig('pctm.jpg')


def pitch_range(sample):

    lowest_pitch = 999
    highest_pitch = 0
    for elem in sample:
        if NOTE_ON in elem:
            cur_pitch_high = int(elem.split('=')[1][-1]) * len(PITCHES) + \
                             PITCHES.index(get_pitch_name_without_octave(elem))

            if cur_pitch_high > highest_pitch:
                highest_pitch = cur_pitch_high
            if cur_pitch_high < lowest_pitch:
                lowest_pitch = cur_pitch_high

    return highest_pitch - lowest_pitch


def average_pitch_interval(sample):
    pass


def note_count(sample):
    pass


def average_inter_onset_interval(sample):
    pass


def note_length_histogram(sample):
    pass


def note_length_transition_matrix(sample):
    pass


def compute_metrics(sample):
    # compute pitch-based metrics
    print(f'Pitch count: {pitch_count(sample)}')
    pitch_class_histogram(sample)
    pitch_class_transition_matrix(sample)
    print(f'Pitch range: {pitch_range(sample)}')
    average_pitch_interval(sample)

    # compute rhythm-based metrics
    note_count(sample)
    average_inter_onset_interval(sample)
    note_length_histogram(sample)
    note_length_transition_matrix(sample)


if __name__ == '__main__':
    file_list, parser = get_bach_chorales()

    pieces = extract_notes(file_list=file_list[:5],
                           parser=parser,
                           )

    print(pieces[0])

    for piece in pieces:
        compute_metrics(piece)
        break
