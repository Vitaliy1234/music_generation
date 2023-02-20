import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


class SelfAttention(nn.Module):
    """SelfAttention class"""
    def __init__(self, input_dim: int, da: int, r: int) -> None:
        """Instantiating SelfAttention class
        Args:
            input_dim (int): dimension of input, eg) (batch_size, seq_len, input_dim)
            da (int): the number of features in hidden layer from self-attention
            r (int): the number of aspects of self-attention
        """
        super(SelfAttention, self).__init__()
        self._ws1 = nn.Linear(input_dim, da, bias=False)
        self._ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_mat = F.softmax(self._ws2(torch.tanh(self._ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class SAN(nn.Module):
    def __init__(self, num_of_dim, vocab_size, embedding_size, r, lstm_hidden_dim=128, da=128, hidden_dim=256) -> None:
        super(SAN, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_size)
        self._bilstm = nn.LSTM(embedding_size, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self._attention = SelfAttention(2 * lstm_hidden_dim, da, r)
        self._classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim * r, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_of_dim)
        )

    def forward(self, x: torch.Tensor):
        fmap = self._embedding(x)
        outputs, hc = self._bilstm(fmap)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score

    def _get_attention_weight(self, x):
        fmap = self._embedding(x)
        outputs, hc = self._bilstm(fmap)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score, attn_mat


def start():
    n_bar_window = 2

    model_file = f'gpt2model_{n_bar_window}_bars'
    tokenizer_path = os.path.join(model_file, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size = tokenizer.vocab_size
    embedding_size = 300

    model = SAN(
        r=14,  # wtf?
        num_of_dim=2,  # num of classes
        vocab_size=vocab_size,  # num of "words" in vocabulary
        embedding_size=embedding_size  # size of embedding
    )

    print(model)
    # data = pd.read_csv('../data/emotion_music/emotion_annotation/verified_annotation.csv')
    # print(data['toptag_eng_verified'].value_counts())


if __name__ == '__main__':
    start()
