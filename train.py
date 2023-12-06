import torch
from torch.utils.data import Dataset, DataLoader

import random

from config import *
from model import GPT

with open("Data/war_and_peace.txt", "r", encoding='utf-8') as f:
    text = f.read()

char_set = sorted(list(set(text)))
VOCAB_SIZE = len(char_set)

word_index = {char: ind for ind, char in enumerate(char_set)}
index_word = {ind: char for ind, char in enumerate(char_set)}
encoder = lambda x: [word_index.get(i, len(word_index)) for i in x]
decoder = lambda x: ''.join([index_word.get(ind, "<OOV>") for ind in x])

data = torch.tensor(encoder(text), dtype=torch.long)

train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


def get_batch(split):
    data = (train_data if split == "train" else test_data)
    start_ind = random.randint(0, len(data) - CHUNK_SIZE - 1)
    x = data[start_ind: start_ind + CHUNK_SIZE]
    y = data[start_ind + 1: start_ind + CHUNK_SIZE + 1]
    return x, y


class GPTDataset(Dataset):
    def __init__(self, generator, length):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return next(self.generator)


def train_generator():
    while True:
        yield get_batch('train')


def test_generator():
    while True:
        yield get_batch('test')


train_dataset = GPTDataset(generator=train_generator(), length=EVAL_ITERS * BATCH_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)


def train():
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpt = GPT(num_layers=NUM_LAYERS,
              vocab_size=VOCAB_SIZE,
              num_heads=NUM_HEADS,
              seq_length=CHUNK_SIZE,
              hidden_dim=HIDDEN_DIM,
              ff_dim=FF_DIM,
              dropout=DROPOUT)
    gpt = gpt.to(device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, loss = gpt(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss, epoch)
        if (epoch > 0) and (epoch % 10 == 0):
            torch.save(gpt.state_dict(), f"Models/trained_model_after_{epoch}_epochs.pth")
        torch.save(gpt.state_dict(), "Models/FINAL_MODEL.pth")

