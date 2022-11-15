import os
import numpy as np
import pypianoroll


def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.zeros(f['shape'], np.bool_)
        data[[x for x in f['nonzero']]] = True
    return data


def parse_npz(filename):
    multitrack = pypianoroll.Multitrack(filename)
    print(multitrack.tracks)
    # pypianoroll.write('cur_melody.mid', multitrack)


if __name__ == '__main__':
    flg = False

    for cur_dir, list_dir, list_files in os.walk('..\\datasets\\lpd_cleansed'):
        for cur_file in list_files:
            if cur_file.endswith('.npz'):
                full_path = os.path.join(cur_dir, cur_file)
                print(full_path)
                m = pypianoroll.Multitrack(os.path.join(cur_dir, cur_file))
                m.write('./test.mid')
                # multitrack = pypianoroll.Multitrack(cur_file)
                # pypianoroll.plot_multitrack(multitrack,)
                # pm = multitrack.to_pretty_midi()
                # print(pm.get_piano_roll())
                flg = True
                break

        if flg:
            break

