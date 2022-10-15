from music21 import note, chord

from data_preparation import get_bach_chorales, extract_notes


def pitch_count(sample):
    pass


def pitch_class_histogram(sample):
    pass


def pitch_class_transition_matrix(sample):
    pass


def pitch_range(sample):
    pass


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
    pitch_count(sample)
    pitch_class_histogram(sample)
    pitch_class_transition_matrix(sample)
    pitch_range(sample)
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

    compute_metrics(pieces)
