import string

class Tokenizer:
    def __init__(self, original_vocab=string.ascii_lowercase + ' ', intermediate_vocab=string.ascii_lowercase, pad_token='<PAD>'):
        self.original_vocab = original_vocab
        self.original_vocab_size = len(original_vocab)

        self.intermediate_chars = intermediate_vocab
        self.intermediate_vocab_size = len(intermediate_vocab)
        
        self.pad_token = pad_token
        self.pad_token_idx = self.original_vocab_size  # The next index after all regular characters
        
        # Mapping for original characters
        self.original_char_to_idx = {char: idx for idx, char in enumerate(self.original_vocab)}
        self.original_idx_to_char = {idx: char for char, idx in self.original_char_to_idx.items()}
        
        # Compressed characters
        self.intermediate_char_to_idx = {char: idx for idx, char in enumerate(self.intermediate_chars)}
        self.intermediate_idx_to_char = {idx: char for idx, char in self.intermediate_char_to_idx.items()}

        # Add the <PAD> token to the vocabulary
        self.original_char_to_idx[pad_token] = self.pad_token_idx
        self.original_idx_to_char[self.pad_token_idx] = self.pad_token
        self.original_vocab_size += 1  # Update vocab size to account for the <PAD> token

    def clean_text(self, text):
        """Removes characters not in the defined vocabulary (lowercase ASCII + space)."""
        return ''.join(char.lower() for char in text if char.lower() in self.original_vocab)

    def encode(self, text):
        """Convert text into token indices."""
        return [self.original_char_to_idx[char] for char in text if char in self.original_char_to_idx]

    def decode(self, indices):
        """Convert token indices back into text, with <PAD> tokens represented."""
        return ''.join(self.original_idx_to_char[idx] if idx != self.pad_token_idx else '<PAD>' for idx in indices)

    def create_word_dictionary(self, text):
        """Create a set of unique words from cleaned text."""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()  # Split by spaces
        word_dict = {word: idx for idx, word in enumerate(set(words))}  # Create a dictionary of unique words
        return word_dict

# Example usage
tokenizer = Tokenizer()

