import music21
from music21 import corpus


def load_music():
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream'):
        notes = []
        track_txt = []
        score.show()
        # read parts
        for part in score.parts:
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
                            # print(f'Time signature: {elem_measure}')
                            pass
                        elif isinstance(elem_measure, music21.note.Note):
                            if elem_measure.isRest:
                                print('rest')
                                bar_txt.append(f'TIME_SHIFT={elem_measure.duration.quarterLength}')
                                # notes.append(elem_measure.name)
                            else:
                                # print(f'duration: {elem_measure.duration}')
                                print(f'Name: {elem_measure.nameWithOctave}')
                                bar_txt.append(f'NOTE_ON={elem_measure.nameWithOctave}')
                                bar_txt.append(f'TIME_SHIFT={elem_measure.duration.quarterLength}')
                                bar_txt.append(f'NOTE_OFF={elem_measure.nameWithOctave}')
                                pitch = elem_measure.beat
                                notes.append(elem_measure)

                        else:
                            # print(type(elem_measure))
                            pass
                    track_txt.append(bar_txt)
                    track_txt.append('BAR_END')
                else:
                    # print(type(elem_part))
                    pass
            print(track_txt)
            break

        # intervals = []
        # for i in range(1, len(notes)):
        #     intervals.append(music21.interval.Interval(notes[i - 1], notes[i]))
        break


if __name__ == '__main__':
    load_music()
