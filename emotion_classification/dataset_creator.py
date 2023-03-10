from pathlib import Path
import os

import pandas as pd

from preprocessing.preprocess_midi import preprocess_music21
from helpers import logging


logger = logging.create_logger("datasetcreator")


class DatasetCreator:
    def __init__(self, dataset_path, raw_midi_dir, annotations):
        """
        Initializing of dataset creator
        :param dataset_path: path to the dataset
        :param raw_midi_dir: path to directory with midis
        :param annotations: file with emotion annotations
        """
        self.dataset_path = dataset_path
        self.raw_midi_dir = raw_midi_dir
        self.annotations = annotations

    def create(self, datasets_path, overwrite=False):
        """
        Creating dataset for model
        :param overwrite: overwrite dataset if it's already exist
        :param datasets_path: path to directory with datasets
        :return:
        """
        Path(datasets_path).mkdir(exist_ok=True)

        dataset_path = os.path.join(datasets_path, self.dataset_path)

        if os.path.exists(dataset_path) and overwrite is False:
            logger.info("Dataset already exists.")
            return

        Path(dataset_path).mkdir(exist_ok=True)

        labels = pd.read_csv(self.annotations)
        for emotion in labels['toptag_eng_verified'].unique():
            cur_files = labels[labels['toptag_eng_verified'] == emotion]['fname'].values()

            songs_data_train, songs_data_valid = preprocess_music21(cur_files)

