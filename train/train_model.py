import os
import random
import tqdm

from typing import Dict

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

# TODO: fix paths while importing
from train_conf import MusicTrainerConf


class MusicTrainer:
    def __init__(self, config: MusicTrainerConf):
        self.config = config

    def train_preparation(self):
        # tokenizer creating
        tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.config.tokenizer_path)
        pretrained_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # model creating
        model_config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.token_to_id('[PAD]'),
            n_head=self.config.n_head,
            n_layer=self.config.n_layer,
            n_embd=self.config.n_embd,
            n_positions=self.config.n_positions,
            n_ctx=self.config.n_ctx
        )

        model = GPT2LMHeadModel(model_config)

        # preparing dataset
        dataset_train = TokenSequenceDataset(
            tokenizer=pretrained_tokenizer,
            dataset_paths=self.config.dataset_train_files,
            block_size=self.config.pad_length,
        )

        # Prepare the validation dataset.
        print("Preparing validate dataset...")
        dataset_valid = TokenSequenceDataset(
            tokenizer=pretrained_tokenizer,
            dataset_paths=self.config.dataset_validate_files,
            block_size=self.config.pad_length,
        )

        # Prepare data collator.
        data_collator = DataCollatorWithPadding(
            tokenizer=pretrained_tokenizer,
            padding="max_length",
            max_length=self.config.pad_length
        )

        # Create the trainer.
        print("Creating trainer...")
        training_args = TrainingArguments(
            output_dir=os.path.join(output_path),
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            num_train_epochs=self.config.epochs,
            per_gpu_train_batch_size=self.config.batch_size,
            save_steps=1_000,
            save_total_limit=2,
            prediction_loss_only=False,
            logging_strategy="steps",
            logging_dir=os.path.join(output_path, "logs"),
            load_best_model_at_end=True,
            save_strategy="steps"
        )
        trainer = CustomLossTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid
        )


class TokenSequenceDataset(Dataset):

    def __init__(self, tokenizer, dataset_paths, block_size, simulate=False):

        pad_token_id = tokenizer.encode("[PAD]")[0]
        unk_token_id = tokenizer.encode("[UNK]")[0]

        # Read all lines from all files.
        lines = []
        for dataset_path in dataset_paths:
            assert os.path.isfile(dataset_path), f"Input file path {dataset_path} not found"
            lines += open(dataset_path, "r").readlines()

        # In simulation just use a few samples.
        if simulate:
            random.shuffle(lines)
            lines = lines[:10]

        # Turn lines into training examples. Also gather some statistics.
        self.examples = []
        unknown_tokens_set = []
        unknown_tokens = []
        tokens_count = 0
        unknown_token_lines_count = 0
        too_long_lines_count = 0
        encoded_lengths = []
        for line in tqdm(lines):

            #Skip empty lines.
            line = line.strip()
            if line == "":
                continue

            # Encode the line.
            encoded_line = tokenizer.encode(line)
            encoded_lengths += [len(encoded_line)]
            tokens_count += len(encoded_line)

            # Create a warning about unknown tokens. And then skip the line.
            if unk_token_id in encoded_line:
                index = encoded_line.index(unk_token_id)
                token = tokenizer.decode(encoded_line[index])
                token = line.split()[index]

                if token not in unknown_tokens_set:
                    unknown_tokens_set += [token]

                unknown_tokens += [token]
                unknown_token_lines_count += 1
                continue

            # Skip sequence if it is too long.
            if len(encoded_line) > block_size:
                too_long_lines_count += 1
                continue

            # Pad and truncate.
            tensor = np.full((block_size,), pad_token_id, dtype=np.long)
            tensor[:len(encoded_line)] = encoded_line
            assert len(tensor) == block_size

            self.examples += [{
                "input_ids": torch.tensor(tensor, dtype=torch.long),
                "labels": torch.tensor(tensor, dtype=torch.long)
            }]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
