import io
import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification,
                          PreTrainedTokenizerFast)
from sklearn.metrics import classification_report, accuracy_score

from music_midi_dataset import MidiMusicDataset


def start():
    set_seed(123)
    epochs = 4
    batch_size = 8
    max_length = 60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'gpt2'
    labels_ids = {'cheerful': 0, 'tense': 1}
    n_labels = len(labels_ids)

    train_midi_data_dir = '../data/music_midi/emotion_midi_text/train'
    test_midi_data_dir = '../data/music_midi/emotion_midi_text/test'
    path_tokenizer = 'tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    classes = ['cheerful', 'tense']
    pad_length = 128
    learning_rate = 0.001

    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                          config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)

    training_data = MidiMusicDataset(midi_data_dir=train_midi_data_dir,
                                     classes=classes,
                                     tokenizer=tokenizer,
                                     block_size=pad_length)
    test_data = MidiMusicDataset(midi_data_dir=test_midi_data_dir,
                                 classes=classes,
                                 tokenizer=tokenizer,
                                 block_size=pad_length)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # default is 1e-8.
                      )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(train_dataloader, model, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, model, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
        train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)


def train(dataloader, model, optimizer_, scheduler_, device_):
    predictions_labels = []
    true_labels = []

    total_loss = 0

    model.train()

    for batch in tqdm(dataloader):
        # print(batch)
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, model, device_):
    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


if __name__ == '__main__':
    start()
