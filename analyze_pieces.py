from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd

from music21 import note, chord

from data_preparation import get_bach_chorales, extract_notes, NOTE_ON, TIME_SHIFT


PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B-', 'B']
NOTES = [4, 2, 1, 0.5, 0.25, 3, 1.5, 0.75, 0.375, 1.333, 0.667, 0.167]


def get_pitch_name_without_octave(pitch):
    return ''.join(pitch.split('=')[1][:-1])


def get_pitch_height(pitch):
    return int(pitch.split('=')[1][-1]) * len(PITCHES) + \
           PITCHES.index(get_pitch_name_without_octave(pitch))


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
    """
    Counts the interval between the lowest and the highest pitches
    :param sample: music
    :return: number of semi-tons between the lowest and the highest pitches
    """
    lowest_pitch = 999
    highest_pitch = 0
    for elem in sample:
        if NOTE_ON in elem:
            cur_pitch_high = get_pitch_height(elem)

            if cur_pitch_high > highest_pitch:
                highest_pitch = cur_pitch_high
            if cur_pitch_high < lowest_pitch:
                lowest_pitch = cur_pitch_high

    return highest_pitch - lowest_pitch


def average_pitch_interval(sample):
    """
    Counts average value of the interval between two consecutive pitches in semi-tones
    :param sample: music
    :return: Average value of the interval between two consecutive pitches in semi-tones
    """
    prev_pitch = ''

    sum_interval = 0
    count_intervals = 0

    for elem in sample:
        if NOTE_ON in elem:
            cur_pitch = get_pitch_height(elem)

            if prev_pitch == '':
                prev_pitch = cur_pitch
                continue

            sum_interval += abs(cur_pitch - prev_pitch)
            count_intervals += 1
            prev_pitch = cur_pitch

    return sum_interval / count_intervals


def note_count(sample):
    """
    Counts number of used notes
    :param sample: music
    :return: number of used notes
    """
    notes = set()

    for elem in sample:
        if TIME_SHIFT in elem:
            notes.add(elem)

    return len(notes)


def average_inter_onset_interval(sample):
    """
    Counts average inter onset interval (time between consecutive notes)
    :param sample: music
    :return: average inter onset interval
    """
    sum_time_interval = 0
    count_time_interval = 0

    for elem in sample:
        if TIME_SHIFT in elem:
            sum_time_interval += float(elem.split('=')[1])
            count_time_interval += 1

    return sum_time_interval / count_time_interval


def note_length_histogram(sample):
    # notes_dict = {cur_note: 0 for cur_note in NOTES}
    notes_dict = defaultdict(float)

    for elem in sample:
        if TIME_SHIFT in elem:
            notes_dict[float(elem.split('=')[1])] += 1

    pitches_df = pd.DataFrame(notes_dict.values(), index=notes_dict.keys(), columns=['note_count'])
    pitches_df.sort_index(inplace=True)

    plt.style.use('seaborn')
    plt.figure(figsize=(6, 5))
    plt.title('Nite class histogram')
    sns.barplot(x=pitches_df.index, y=pitches_df['note_count'])
    plt.xlabel('Note name')
    plt.ylabel('Note count')
    plt.savefig('note_class_histogram.jpg')


def note_length_transition_matrix(sample):
    notes_arr = np.zeros((len(NOTES), len(NOTES)))
    pass


def compute_metrics(sample):
    # compute pitch-based metrics
    print(f'Pitch count: {pitch_count(sample)}')
    pitch_class_histogram(sample)
    pitch_class_transition_matrix(sample)
    print(f'Pitch range: {pitch_range(sample)}')
    print(f'Average pitch interval: {average_pitch_interval(sample)}')

    # compute rhythm-based metrics
    print(f'Number of used notes: {note_count(sample)}')
    print(f'Average inter-onset interval: {average_inter_onset_interval(sample)}')
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
