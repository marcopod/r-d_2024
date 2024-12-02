import torch
import random
import torch.nn.functional as F  
from model_architecture import obfuscator, deobfuscator
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

# Specify the file paths to save your models
encoder_path = "./code/models/encoder_model_128_9.pth"
decoder_path = "./code/models/decoder_model_128_9.pth"
import os
if not os.path.exists(encoder_path):
    print(f"---------------------------------------File not found")
    print("Current working directory:", os.getcwd())


# Load the obfuscator model safely
obfuscator.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))

# Load the deobfuscator model safely
deobfuscator.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

def demonstrate_compression(autoencoder1, autoencoder2, text):
    autoencoder1.eval()
    autoencoder2.eval()
    
    with torch.no_grad():
        # Tokenize and prepare input
        tokenized = tokenizer.encode(text)
        print(seq_len)
        if len(tokenized) < seq_len:
            tokenized += [tokenizer.original_char_to_idx[tokenizer.pad_token]] * (seq_len - len(tokenized))  # Pad to seq_len using <PAD>
        input_seq = torch.tensor(tokenized[:seq_len], dtype=torch.long).unsqueeze(0).to(device)
        print(input_seq.shape)
        

        # Step 1: Compress using autoencoder1 (Encoder)
        compressed_output = autoencoder1(input_seq)

        # Step 2: Take argmax over the output to get discrete token indices
        compressed_indices = torch.argmax(compressed_output, dim=-1)  # Shape: (batch_size, output_length)
        compressed_chars = ''.join([tokenizer.original_idx_to_char[idx.item()] for idx in compressed_indices[0]])

        # Step 3: Convert argmax indices back to embeddings or one-hot vectors (optional based on your deobfuscator)
        # Assuming we can pass the indices directly if deobfuscator is equipped to handle them:
        compressed_indices_one_hot = F.one_hot(compressed_indices, num_classes=autoencoder1.output_vocab_size).float()
        # print(compressed_indices_one_hot)

        # Step 4: Decompress using autoencoder2 (Decoder), passing the argmax indices in one-hot format or directly
        reconstructed_output = autoencoder2(compressed_indices_one_hot)

        # Step 5: Take argmax on the deobfuscator output to get the predicted token indices
        clamped_output = torch.clamp(torch.argmax(reconstructed_output, dim=-1), 0, tokenizer.original_vocab_size - 1)

        # Convert indices back to characters
        reconstructed_chars = ''.join([
            tokenizer.original_idx_to_char[idx.item()] if idx.item() != tokenizer.original_char_to_idx[tokenizer.pad_token] else tokenizer.pad_token 
            for idx in clamped_output[0]
        ])

        return compressed_chars, reconstructed_chars

# Example usage
#input_text = tokenizer.decode(progressive_train_sequences[62])
input_text = "hello world "
compressed, reconstructed = demonstrate_compression(obfuscator, deobfuscator, input_text)
print(f"Original: {input_text}")
print(f"Obfuscated ({len(compressed)} chars): {compressed}")
print(f"Deobfuscated: {reconstructed}")

