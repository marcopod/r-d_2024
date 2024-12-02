import torch
import torch.nn as nn
from tokenizer import tokenizer

# Hyperparameters
embed_dim = 64
hidden_dim = 64
seq_len = 16 # Input and reconstruction lenght
optimized_len = 128 # Optimized text lenght
num_epochs = 1
train_rows = 1  # Number of articles of the BBC data to train on
test_rows = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Obfuscator(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, hidden_dim, input_len, output_length):
        super(Obfuscator, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.projection_out_vocab = nn.Linear(embed_dim, output_vocab_size)
        self.projection_out_len = nn.Linear(input_len, output_length)
        self.output_vocab_size = output_vocab_size
        
    def forward(self, x):
        # x.shape: batch_size, seq_len
        # embedded.shape: batch_size, seq_len, embed_dim
        embedded = self.embedding(x)
        # encoded_output.shape: batch_size, seq_len, hidden_dim
        encoded_output, _ = self.encoder(embedded)
        
        # decoded_output.shape: batch_size, seq_len, embed_dim
        decoded_output, _ = self.decoder(encoded_output)
        
        # decoded_output.shape: batch_size, seq_len, intermediate_vocab_size
        projected_out_vocab_size = self.projection_out_vocab(decoded_output)
        
        # decoded_output.shape: batch_size, intermediate_seq_len, vocab_size
        projected_out_len = self.projection_out_len(projected_out_vocab_size.transpose(1, 2)).transpose(1, 2)
        # decoded_output.shape: batch_size, intermediate_seq_len, vocab_size
        logits = projected_out_len
        
        return logits
    
class Deobfuscator(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, hidden_dim, input_len, output_length):
        super(Deobfuscator, self).__init__()
        self.embedding = nn.Linear(input_vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.projection_out_vocab = nn.Linear(embed_dim, output_vocab_size)
        self.projection_out_len = nn.Linear(input_len, output_length)
        self.output_vocab_size = output_vocab_size        

    def forward(self, x):
        # x.shape: batch_size, intermediate_seq_len, intermediate_vocab_size
        # embedded.shape: batch_size, intermediate_seq_len, embed_dim
        embedded = self.embedding(x)
        # encoded_output.shape: batch_size, intermediate_seq_len, hidden_dim
        encoded_output, _ = self.encoder(embedded)
        
        # encoded_output.shape: batch_size, intermediate_seq_len, hidden_dim
        decoded_output, _ = self.decoder(encoded_output)
        
        # encoded_output.shape: batch_size, intermediate_seq_len, vocab_size
        projected_out_vocab_size = self.projection_out_vocab(decoded_output)
        
        # encoded_output.shape: batch_size, seq_len, vocab_size
        projected_out_len = self.projection_out_len(projected_out_vocab_size.transpose(1, 2)).transpose(1, 2)
        logits = projected_out_len
        return logits
    
    
# First autoencoder: compression (input length: seq_length, output length: optimized_len)
obfuscator = Obfuscator(
    input_vocab_size=tokenizer.original_vocab_size,
    output_vocab_size=tokenizer.intermediate_vocab_size,  # All printable ASCII characters
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    input_len=seq_len,
    output_length=optimized_len
).to(device)

# Second autoencoder: decompression (input length: optimized_len, output length: seq_length)
deobfuscator = Deobfuscator(
    input_vocab_size=tokenizer.intermediate_vocab_size,  # All printable ASCII characters
    output_vocab_size=tokenizer.original_vocab_size,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    input_len=optimized_len,
    output_length=seq_len
).to(device)
