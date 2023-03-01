import os
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer

from classifier import SAN
from music_midi_dataset import MidiMusicDataset


def create_tokenizer(data_files, tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(files=data_files, trainer=trainer)
    tokenizer.save(tokenizer_path)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for cur_obj in dataloader:
            input_ids, labels = cur_obj['input_ids'], cur_obj['labels']
            pred = model(input_ids)
            test_loss += loss_fn(pred, labels.reshape(-1)).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, cur_obj in enumerate(dataloader):
        input_ids, labels = cur_obj['input_ids'], cur_obj['labels']

        # forward pass
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels.reshape(-1))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def start():
    train_midi_data_dir = '../data/music_midi/emotion_midi_text/train'
    test_midi_data_dir = '../data/music_midi/emotion_midi_text/test'
    path_tokenizer = 'tokenizer.json'
    output_path = 'classifier_model'
    Path(output_path).mkdir(exist_ok=True)

    learning_rate = 0.01
    num_epoch = 5
    batch_size = 8

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size = tokenizer.vocab_size
    embedding_size = 300
    pad_length = 128

    classes = ['cheerful', 'tense']

    model = SAN(
        r=14,  # wtf?
        num_of_dim=len(classes),  # num of classes
        vocab_size=vocab_size,  # num of "words" in vocabulary
        embedding_size=embedding_size  # size of embedding
    )

    training_data = MidiMusicDataset(midi_data_dir=train_midi_data_dir,
                                     classes=classes,
                                     tokenizer=tokenizer,
                                     block_size=pad_length)
    test_data = MidiMusicDataset(midi_data_dir=test_midi_data_dir,
                                 classes=classes,
                                 tokenizer=tokenizer,
                                 block_size=pad_length)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer)
        test_loop(test_dataloader, model, criterion)
    print("Done!")
    # data = pd.read_csv('../data/emotion_music/emotion_annotation/verified_annotation.csv')
    # print(data['toptag_eng_verified'].value_counts())


if __name__ == '__main__':
    # Creating tokenizer
    # text_midi_files = []
    # root_dataset_dir = '../data/music_midi/emotion_midi_text'
    # path_tokenizer = 'tokenizer.json'
    #
    # for cur_dir, dirs, files in os.walk(root_dataset_dir):
    #     for file in files:
    #         if file.endswith('.txt'):
    #             text_midi_files.append(os.path.join(cur_dir, file))
    #
    # create_tokenizer(data_files=text_midi_files, tokenizer_path=path_tokenizer)

    start()
