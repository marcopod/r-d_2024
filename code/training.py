import torch
import torch.nn as nn
import torch.nn.functional as F  
from tqdm import tqdm

from dataset import TextDataset
from torch.utils.data import DataLoader
from model_architecture import obfuscator, deobfuscator

# Hyperparameters
embed_dim = 64
hidden_dim = 64
seq_len = 16 # Input and reconstruction lenght
intermediate_len = 128 # Intermediate text lenght
num_epochs = 15
train_sequences = 30_000  # Number of articles of the BBC data to train on
test_sequences = 10_000
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the dataset with generated sequences
text_dataset = TextDataset(seq_len=seq_len, train_sequences=train_sequences, test_sequences=test_sequences)

# Create DataLoaders for training and evaluation
train_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)
eval_dataloader = DataLoader(text_dataset.get_eval_data(), batch_size=32, shuffle=True)

# Define the loss function and optimizers
criterion = nn.CrossEntropyLoss()


# Run training multiple times and save the losses for each run
num_runs = 1  # Number of times to repeat the training loop

print(f"\n--- Starting Training Run ---")
        
encoder_optimizer = torch.optim.Adam(obfuscator.parameters(), lr=lr)
decoder_optimizer = torch.optim.Adam(deobfuscator.parameters(), lr=lr)
    
# Lists to store losses and accuracies for the current run
train_losses = []
eval_accuracies = []
eval_one_hot_accuracies = []
def train_model(encoder, decoder, train_dataloader, eval_dataloader, criterion, encoder_optimizer, decoder_optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        total_train_loss = 0

        print(f"Epoch {epoch+1}/{num_epochs}")
        encoder.train()
        decoder.train()
        train_loader_tqdm = tqdm(train_dataloader, desc="Training", leave=False)
        for sequences in train_loader_tqdm:
            sequences = sequences.to(device).long()

            # Compression and decompression steps
            compressed_sequences = encoder(sequences)
            if torch.rand(1).item() < 0.9:
                compressed_indices = torch.argmax(compressed_sequences, dim=-1)
                compressed_sequences = F.one_hot(compressed_indices, num_classes=encoder.output_vocab_size).float()

            reconstructed_sequences = decoder(compressed_sequences).transpose(1, 2)
            loss = criterion(reconstructed_sequences, sequences)

            # Optimization
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=0.5)
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_train_loss += loss.item() * sequences.size(0)
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        average_train_loss = total_train_loss / len(train_dataloader.dataset)
        train_losses.append(average_train_loss)

        # Validation accuracy calculation
        encoder.eval()
        decoder.eval()
        total_correct_eval = 0
        total_tokens_eval = 0
        total_correct_eval_one_hot = 0

        eval_loader_tqdm = tqdm(eval_dataloader, desc="Validating", leave=False)
        with torch.no_grad():
            for sequences in eval_loader_tqdm:
                sequences = sequences.to(device).long()
                compressed_sequences = encoder(sequences)
                reconstructed_sequences = decoder(compressed_sequences).transpose(1, 2)
                predicted_tokens_eval = torch.argmax(reconstructed_sequences, dim=1)
                total_correct_eval += (predicted_tokens_eval == sequences).sum().item()
                total_tokens_eval += sequences.numel()

                compressed_indices = torch.argmax(compressed_sequences, dim=-1)
                compressed_one_hot = F.one_hot(compressed_indices, num_classes=encoder.output_vocab_size).float()
                reconstructed_sequences_one_hot = decoder(compressed_one_hot).transpose(1, 2)
                predicted_tokens_eval_one_hot = torch.argmax(reconstructed_sequences_one_hot, dim=1)
                total_correct_eval_one_hot += (predicted_tokens_eval_one_hot == sequences).sum().item()

        eval_accuracy = total_correct_eval / total_tokens_eval
        eval_accuracies.append(eval_accuracy)
        eval_one_hot_accuracy = total_correct_eval_one_hot / total_tokens_eval
        eval_one_hot_accuracies.append(eval_one_hot_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {average_train_loss:.4f}, Validation Accuracy = {eval_accuracy:.4f}, One-Hot Validation Accuracy = {eval_one_hot_accuracy:.4f}")

    return train_losses, eval_accuracies, eval_one_hot_accuracies
    # Run the training model and save metrics
train_losses, eval_accuracies, eval_one_hot_accuracies = train_model(obfuscator, deobfuscator, train_dataloader, eval_dataloader, criterion, encoder_optimizer, decoder_optimizer, num_epochs)