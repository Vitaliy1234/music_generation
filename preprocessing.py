import music21
from music21 import corpus


def load_music():
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream'):
        notes = []
        # read parts
        for part in score.parts:
            # read current part
            for elem_part in part:
                if isinstance(elem_part, music21.instrument.Instrument):
                    # print(f'Instrument: {elem_part}')
                    pass
                elif isinstance(elem_part, music21.stream.base.Measure):
                    # print(f'Measure: {elem_part}')

                    # read measure
                    for elem_measure in elem_part:
                        if isinstance(elem_measure, music21.key.Key):
                            # print(f'Key: {elem_measure}')
                            pass
                        elif isinstance(elem_measure, music21.meter.base.TimeSignature):
                            # print(f'Time signature: {elem_measure}')
                            pass
                        elif isinstance(elem_measure, music21.note.Note):
                            if elem_measure.isRest:
                                print('rest')
                                # notes.append(elem_measure.name)
                            else:
                                # print(f'duration: {elem_measure.duration}')
                                print(f'Name: {elem_measure.nameWithOctave}')
                                # print(elem_measure)
                                notes.append(elem_measure)

                        else:
                            # print(type(elem_measure))
                            pass
                else:
                    # print(type(elem_part))
                    pass

        intervals = []
        for i in range(1, len(notes)):
            intervals.append(music21.interval.Interval(notes[i - 1], notes[i]))
        break


if __name__ == '__main__':
    load_music()
