import os

from miditok import REMI, CPWord
from miditoolkit import MidiFile

from transformers import DataCollatorWithPadding
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from GPT2RGA import EPianoDataset, GPTConfig, VOCAB_SIZE, dim_feedforward, GPT, get_device, LR_DEFAULT_START, \
    LrStepTracker, d_model, SCHEDULER_WARMUP_STEPS, TOKEN_PAD, ADAM_BETA_1, ADAM_BETA_2, ADAM_EPSILON, epochs, train, \
    eval_model


def miditok_test():
    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)

    # Creates the tokenizer and loads a MIDI
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
    # tokenizer = CPWord(pitch_range, beat_res, nb_velocities, additional_tokens)
    midi = MidiFile(os.path.join('data', 'optimus_dataset.mid'))

    tokens = tokenizer.midi_to_tokens(midi)

    number_of_batches = 8
    n_workers = 6
    max_seq = 1024
    random_seq = True

    print('=' * 50)
    print('Prepping INTs datasets...')

    train_data = []

    train_data.extend(tokens[0][:20000])

    val_dataset = train_data[:int(len(train_data) * 0.5)]
    test_dataset = train_data[:int(len(train_data) * 0.5)]

    train_list = train_data
    val_list = val_dataset
    test_list = []
    print('=' * 50)

    print('Processing INTs datasets...')
    train_dataset = EPianoDataset(train_list, max_seq, random_seq)
    val_dataset = EPianoDataset(val_list, max_seq)
    test_dataset = EPianoDataset(test_list, max_seq)
    print('=' * 50)

    print('Loading INTs datasets...')
    batch_size = number_of_batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers)
    print('=' * 50)

    print('Total INTs in the dataset', len(train_data))
    print('Total unique INTs in the dataset', len(set(train_data)))
    print('Max INT in the dataset', max(train_data))
    print('Min INT in the dataset', min(train_data))
    print('=' * 50)

    print('Checking datasets shapes...')
    print('=' * 50)

    print('Train loader')
    for x, tgt in train_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Validation loader')
    for x, tgt in val_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Test loader')
    for x, tgt in test_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Done! Enjoy! :)')
    print('=' * 50)

    # @title Train

    print('MidiTok Model Trainer')

    config = GPTConfig(VOCAB_SIZE,
                       max_seq,
                       dim_feedforward=dim_feedforward,
                       n_layer=6,
                       n_head=8,
                       n_embd=512,
                       enable_rpr=True,
                       er_len=max_seq)
    model = GPT(config).to(get_device())

    # =====

    init_step = 0
    lr = LR_DEFAULT_START
    lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    train_loss_func = eval_loss_func

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step)

    # ===

    best_eval_acc = 0.0
    best_eval_acc_epoch = -1
    best_eval_loss = float("inf")
    best_eval_loss_epoch = -1
    best_acc_file = 'gpt2_rpr_acc.pth'
    best_loss_file = 'gpt2_rpr_loss.pth'
    loss_train, loss_val, acc_val = [], [], []

    for epoch in range(0, epochs):
        new_best = False

        loss = train(epoch + 1, model, train_loader, train_loss_func, opt, lr_scheduler, num_iters=-1)
        loss_train.append(loss)

        eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)
        loss_val.append(eval_loss)
        acc_val.append(eval_acc)

        if (eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch = epoch + 1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if (eval_loss < best_eval_loss):
            best_eval_loss = eval_loss
            best_eval_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        if (new_best):
            print("Best eval acc epoch:", best_eval_acc_epoch)
            print("Best eval acc:", best_eval_acc)
            print("")
            print("Best eval loss epoch:", best_eval_loss_epoch)
            print("Best eval loss:", best_eval_loss)


if __name__ == '__main__':
    miditok_test()
