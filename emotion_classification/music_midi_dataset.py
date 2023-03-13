import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MidiMusicDataset(Dataset):
    """
    Class for midi music dataset
    """
    def __init__(self, text_midis, labels, tokenizer, block_size):
        """
        Initialization of midi music dataset
        :param text_midis: midis in text format
        :param labels: class labels
        :param tokenizer:
        :param block_size:
        """
        pad_token_id = tokenizer.encode("[PAD]")[0]
        unk_token_id = tokenizer.encode("[UNK]")[0]

        self.examples = []
        self.tokenizer = tokenizer

        classes = list(set(labels))

        for line, label in zip(text_midis, labels):
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
