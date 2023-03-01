import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MidiMusicDataset(Dataset):
    """
    Class for midi music dataset
    """
    def __init__(self, midi_data_dir, classes, tokenizer, block_size):
        """
        Initialization of midi music dataset class
        :param midi_data_dir: name of directory with midis
        :param classes:
        """
        pad_token_id = tokenizer.encode("[PAD]")[0]
        unk_token_id = tokenizer.encode("[UNK]")[0]

        self.examples = []
        self.tokenizer = tokenizer

        lines = []
        labels = []

        for class_name in classes:
            cur_file = os.path.join(midi_data_dir, class_name) + '.txt'
            with open(cur_file, 'r') as midi_text:
                midi_text_lines = midi_text.readlines()

            lines.extend(midi_text_lines)
            labels.extend([class_name] * len(midi_text_lines))

        for line, label in zip(lines, labels):
            line = line.strip()
            if line == "":
                continue

            encoded_line = tokenizer.encode(line)

            tensor_melody = np.full((block_size,), pad_token_id, dtype=np.long)
            tensor_melody[:len(encoded_line)] = encoded_line[:block_size]

            tensor_label = torch.tensor(classes.index(label))

            self.examples += [{
                "input_ids": torch.tensor(tensor_melody, dtype=torch.long),
                "labels": torch.tensor(tensor_label, dtype=torch.long)
            }]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
