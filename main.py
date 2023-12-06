import torch

from model import GPT
from config import *
from train import VOCAB_SIZE, encoder, decoder, char_set


def main():
    gpt = GPT(num_layers=NUM_LAYERS,
              vocab_size=VOCAB_SIZE,
              num_heads=NUM_HEADS,
              seq_length=CHUNK_SIZE,
              hidden_dim=HIDDEN_DIM,
              ff_dim=FF_DIM,
              dropout=DROPOUT)
    gpt.load_state_dict(torch.load("Models/model.pt"))
    while True:
        try:
            number_of_new_chars = input("Enter the number of tokens to generate: ")
            number_of_new_chars = int(number_of_new_chars)
            starting_string = torch.tensor([encoder("\n")])
            for token in gpt.generate_generator(starting_string, number_of_new_chars):
                print(decoder(token[0].tolist()), end='')
        except:
            break

if __name__ == "__main__":
    main()