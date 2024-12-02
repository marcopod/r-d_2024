from tokenizer import tokenizer

import torch
from torch.utils.data import Dataset
import random

class TextDataset(Dataset):
    def __init__(self, seq_len=16, train_sequences=100, test_sequences=20, pad_token_idx=tokenizer.pad_token_idx):
        self.seq_len = seq_len
        self.pad_token_idx = pad_token_idx
        self.word_dict = {}

        # Generate random data for training and evaluation
        train_data, eval_data = self.generate_random_sequences(train_sequences, test_sequences)

        # Flatten the generated data into a single list of tokens for creating sequences
        flat_train_data = [token for sublist in train_data for token in sublist]
        flat_eval_data = [token for sublist in eval_data for token in sublist]

        # Create sequences from the tokenized data, ensuring no words are cut off
        self.train_sequences = self.create_sequences_without_word_cutoff(flat_train_data, self.seq_len)
        self.eval_sequences = self.create_sequences_without_word_cutoff(flat_eval_data, self.seq_len)

        # Generate progressive sequences for training and evaluation
        self.progressive_train_sequences = self.generate_progressive_sequences(self.train_sequences, self.seq_len)
        self.progressive_eval_sequences = self.generate_progressive_sequences(self.eval_sequences, self.seq_len)

        # Prepare PyTorch tensors for training and evaluation data
        self.train_data = [torch.tensor(seq, dtype=torch.long) for seq in self.progressive_train_sequences]
        self.eval_data = [torch.tensor(seq, dtype=torch.long) for seq in self.progressive_eval_sequences]

    def encode(self, text):
        """Encodes text to indices."""
        return [tokenizer.original_char_to_idx.get(char, self.pad_token_idx) for char in text]

    def generate_random_sequences(self, train_sequences, test_sequences):
        """Generate random sequences for training and evaluation datasets."""
        characters = list(tokenizer.original_char_to_idx.keys())
        train_data = []
        eval_data = []

        for _ in range(train_sequences):
            random_text = ''.join(random.choices(characters, k=self.seq_len))
            train_data.append(self.encode(random_text))

        for _ in range(test_sequences):
            random_text = ''.join(random.choices(characters, k=self.seq_len))
            eval_data.append(self.encode(random_text))

        return train_data, eval_data

    def create_sequences_without_word_cutoff(self, tokenized_text, seq_len):
        sequences = []
        current_sequence = []
        current_length = 0

        words = ''.join([tokenizer.original_idx_to_char[idx] for idx in tokenized_text]).split()

        for word in words:
            word_tokenized = self.encode(word) + [tokenizer.original_char_to_idx[' ']]
            word_length = len(word_tokenized)

            if current_length + word_length > seq_len:
                current_sequence += [self.pad_token_idx] * (seq_len - current_length)
                sequences.append(current_sequence)
                current_sequence = word_tokenized
                current_length = word_length
            else:
                current_sequence += word_tokenized
                current_length += word_length

        if current_sequence:
            current_sequence += [self.pad_token_idx] * (seq_len - current_length)
            sequences.append(current_sequence)

        return sequences

    def generate_progressive_sequences(self, sequences, max_len):
        progressive_sequences = []
        pad_value = self.pad_token_idx

        for sequence in sequences:
            pad_found = False
            for i in range(1, max_len + 1):
                if pad_found:
                    break
                padded_sequence = sequence[:i] + [pad_value] * (max_len - i)
                progressive_sequences.append(padded_sequence)
                if pad_value in padded_sequence[:i]:
                    pad_found = True

        return progressive_sequences

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index]

    def get_eval_data(self):
        return self.eval_data