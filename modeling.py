from transformers import pipeline

from data_preparation import extract_notes, get_bach_chorales, BAR_START, BAR_END, PIECE_START, PIECE_END, \
                             TRACK_START, TRACK_END, INSTRUMENT, NOTE_ON, NOTE_OFF, TIME_SHIFT


vocabulary = set()


def tokenize(pieces):
    for piece in pieces:
        one_str = ';'.join(piece)
        one_str = one_str.replace('=', ';')

        one_str_list = one_str.split(';')

        for token in one_str_list:
            vocabulary.add(token)

    print(vocabulary)


def main():
    file_list, parser = get_bach_chorales()

    pieces = extract_notes(file_list[:5], parser)
    tokenize(pieces)

    text_generator = pipeline('text-generation')

    prefix_text = 'The world is'

    print(text_generator("We are very happy to show our transformer library"))


if __name__ == '__main__':
    main()
