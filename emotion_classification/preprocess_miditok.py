import os

from miditok import REMI, CPWord
from miditoolkit import MidiFile

from music21 import converter


def miditok_test(midi_file):
    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)

    # Creates the tokenizer and loads a MIDI
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
    # tokenizer = CPWord(pitch_range, beat_res, nb_velocities, additional_tokens)

    # midi_files = [os.path.join(midi_data_dir, cur_midi_file) for cur_midi_file in os.listdir(midi_data_dir)]
    # midi = converter.parse(midi_files[1])
    # measures = midi.measures(0, midi.measureNumber)

    # tokens = tokenizer.tokenize_midi_dataset(midi_files[:1], output_dir)
    tokens = tokenizer.midi_to_tokens(MidiFile(midi_file))

    # for CP
    # for vocab in tokenizer.vocab:
    #     print(vocab.event_to_token)
    token_to_evnt = tokenizer.vocab.token_to_event

    return ' '.join([token_to_evnt[cur_token] for cur_token in tokens[0]])


if __name__ == '__main__':
    miditok_test(os.path.join('../data', 'v_lesu_elka.mid'))
