import note_seq
from helpers.samplinghelpers import render_token_sequence, token_sequence_to_note_sequence


def visualize_notes(file_txt):
    with open(file_txt, 'r') as hfile:
        token_seq = hfile.read()

    note_seq.note_sequence_to_midi_file(token_sequence_to_note_sequence(token_seq), 'melody.mid')


if __name__ == '__main__':
    visualize_notes('text_repr.txt')
