import music21
from music21 import corpus


def load_music():
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream'):
        for part in score.parts:
            for elem_part in part:
                if isinstance(elem_part, music21.instrument.Instrument):
                    print(f'Instrument: {elem_part}')
                elif isinstance(elem_part, music21.stream.base.Measure):
                    print(f'Measure: {elem_part}')

                    for elem in elem_part:
                        print(type(elem))
                else:
                    print(type(elem_part))

        break


if __name__ == '__main__':
    load_music()
