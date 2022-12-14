import os

from train_conf import MusicTrainerConf
from train_model import MusicTrainer


CUR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def train():
    dir_up = os.path.split(CUR_ABS_PATH)[0]
    train_config = MusicTrainerConf(
        tokenizer_path=os.path.join(dir_up, 'gpt2model_4_bars', 'tokenizer.json'),
        dataset_train_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_train.txt")],
        dataset_validate_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_valid.txt")],
    )


if __name__ == '__main__':
    train()
