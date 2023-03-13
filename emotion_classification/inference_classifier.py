import os
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer

from classifier import SAN
from music_midi_dataset import MidiMusicDataset

from data_preparation import transpose_text_midi


def create_tokenizer(data_files, tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(files=data_files, trainer=trainer)
    tokenizer.save(tokenizer_path)


def valid_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for cur_obj in dataloader:
            input_ids, labels = cur_obj['input_ids'].to(device), cur_obj['labels'].to(device)
            pred = model(input_ids)
            test_loss += loss_fn(pred, labels.reshape(-1)).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, cur_obj in enumerate(dataloader):
        input_ids, labels = cur_obj['input_ids'].to(device), cur_obj['labels'].to(device)

        # forward pass
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels.reshape(-1))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def get_dataset(annotations_path, midi_texts_path):
    annots = pd.read_csv(annotations_path)
    annots = annots[annots['toptag_eng_verified'].isin(['cheerful', 'tense'])]
    # TODO: вынести объявление классов куда-нибудь отдельно
    classes = {'cheerful': 0,
               'tense': 1}

    annots['toptag_eng_verified'] = annots['toptag_eng_verified'].replace(classes).astype('float')

    midi_texts = []

    for txt_midi in tqdm(annots['fname']):
        # read encoded midi via MMM Encoding in my realization
        with open(os.path.join(midi_texts_path, txt_midi.replace('.mid', '.txt')), 'r') as t_mid:
            midi_texts.append(t_mid.read())

    annots['midi_text'] = midi_texts

    split_index = int(annots.shape[0] * 0.7)
    annots_train = annots[:split_index]
    annots_test = annots[split_index:]

    annots_train['midi_text'] = annots_train['midi_text'].apply(lambda elem: transpose_text_midi(elem, range(12)))
    annots_train = annots_train.explode('midi_text')

    return annots_train, annots_test


def start():
    # midi_data_dir = '/Users/18629082/Desktop/music_generation/data/music_midi/emotion_midi_texts'
    midi_data_dir = r'D:\Диссер_музыка\music_generation\data\music_midi\emotion_midi_texts'
    annotations_path = os.path.join('../data', 'music_midi', 'verified_annotation.csv')
    annots_train, annots_test = get_dataset(annotations_path, midi_data_dir)
    path_tokenizer = 'tokenizer.json'
    output_path = 'classifier_model'
    Path(output_path).mkdir(exist_ok=True)

    learning_rate = 0.001
    num_epoch = 20
    batch_size = 8

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size = tokenizer.vocab_size
    embedding_size = 10
    pad_length = 512

    classes = ['cheerful', 'tense']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SAN(
        r=5,  # wtf?
        num_of_dim=len(classes),  # num of classes
        vocab_size=vocab_size,  # num of "words" in vocabulary
        embedding_size=embedding_size,  # size of embedding
        lstm_hidden_dim=8,
        da=8,
        hidden_dim=32
    )
    model.to(device)

    training_data = MidiMusicDataset(text_midis=annots_train['midi_text'],
                                     labels=annots_train['toptag_eng_verified'],
                                     tokenizer=tokenizer,
                                     block_size=pad_length)
    test_data = MidiMusicDataset(text_midis=annots_test['midi_text'],
                                 labels=annots_test['toptag_eng_verified'],
                                 tokenizer=tokenizer,
                                 block_size=pad_length)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer, device)
        valid_loop(test_dataloader, model, criterion, device)
    print("Done!")
    x = test_data.__getitem__(0)['input_ids']
    x = x.unsqueeze(dim=0)
    x.to('cpu')
    model.to('cpu')
    score, attn_map = model._get_attention_weight(x)

    print(score, attn_map)


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
