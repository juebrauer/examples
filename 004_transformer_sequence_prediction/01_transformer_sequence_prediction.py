"""
Simple encoder-decoder Transformer example

Will learn to map 3 letters of the alphabet to the next 3 letters, e.g.
input: GHI
output: JKL

With this, you can see and learn about all important parts of the Transformer that are involved.

by Jürgen Brauer
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class Tokenizer:
    """
    Maps a sequence of tokens to a sequence of token IDs

    encode: maps tokens to token IDs
    decode: maps token IDs to tokens
    """
    def __init__(self, sequences):
        self.tokens = set({"<pad>", "<eos>", "<start>"})
        for sequence in sequences:
            self.tokens = self.tokens.union(set(sequence))
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, sequence):
        return [self.token_to_id[character] for character in sequence]

    def decode(self, token_ids):
        return [self.id_to_token[idx] for idx in token_ids]


class SequenceDataset(Dataset):
    """
    Provides a PyTorch example dataset with (input, target) pairs
    """
    def __init__(self, input_sequences, target_sequences, pad_token_id=0):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_sequence = self.input_sequences[idx]
        target_sequence = self.target_sequences[idx]
        return (
            torch.tensor(input_sequence, dtype=torch.long),
            torch.tensor(target_sequence, dtype=torch.long),
        )


def generate_square_subsequent_mask(size):
    """
    Provides a look-ahead mask

    This mask is important for learning to map one
    sequence in an autoregressive fashion to another sequence
    token by token.    
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))  # Unzugängliche Werte auf -inf setzen
    return mask


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_length, dropout=0.1):
        super(EncoderDecoderTransformer, self).__init__()
        # Separate Embedding-Schichten für Encoder und Decoder
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)

   
    def forward(self, src, tgt):
        src_seq_length = src.size(1)
        tgt_seq_length = tgt.size(1)

        src = self.encoder_embedding(src) + self.positional_encoding[:, :src_seq_length, :]
        tgt = self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]

        memory = self.encoder(src)
        
        # Generate causal mask
        tgt_mask = generate_square_subsequent_mask(tgt_seq_length).to(tgt.device)

        # Deactivate causual mask in order to see that the Transformer
        # will have severe problems to predict in inference mode.        
        use_causual_mask = True
        if use_causual_mask:
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        else:
            output = self.decoder(tgt, memory)

        logits = self.output_layer(output)
        return logits



def generate_examples_from_alphabet(alphabet):
    """
    Helper function to create example training pairs

    Example: [("ABC", "DEF"), ("BCD", "EFG"), ...]
    """
    examples = []
    for i in range(len(alphabet) - 3):
        input_seq = alphabet[i:i + 3]
        target_seq = alphabet[i + 3:i + 6]
        examples.append((input_seq, target_seq))
    return examples


def generate_predictions(model, tokenizer, sequences, device, max_length):
    """
    Use the trained Transformer to predict an output sequence
    """
    model.eval()
    predictions = {}

    for sequence in sequences:
        input_ids = tokenizer.encode(sequence)
        
        # Direkt den Forward-Pass des Modells verwenden
        src = torch.tensor([tokenizer.token_to_id["<start>"]] + input_ids, dtype=torch.long).unsqueeze(0).to(device)
        generated_ids = [tokenizer.token_to_id["<start>"]]

        for step in range(max_length):
            tgt_tensor = torch.tensor(generated_ids, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(src, tgt_tensor)

            # Wähle das Token mit der höchsten Wahrscheinlichkeit
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            generated_ids.append(next_token)

            if next_token == tokenizer.token_to_id["<eos>"]:                
                break

        predictions[sequence] = tokenizer.decode(generated_ids[1:])

    return predictions



if __name__ == "__main__":

    # Parameters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    batch_size = 4
    d_model = 32
    num_heads = 4
    num_layers = 4
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate examples
    examples = generate_examples_from_alphabet(alphabet)
    input_sequences = [ex[0] for ex in examples]
    target_sequences = [ex[1] for ex in examples]

    tokenizer = Tokenizer([alphabet])

    # Map input and output token sequences
    # to token ID sequences with special <start> and <eos> token IDs
    encoded_inputs = [
        [tokenizer.token_to_id["<start>"]] + tokenizer.encode(seq) + [tokenizer.token_to_id["<eos>"]] for seq in input_sequences
    ]
    encoded_targets = [
        [tokenizer.token_to_id["<start>"]] + tokenizer.encode(seq) + [tokenizer.token_to_id["<eos>"]] for seq in target_sequences
    ]
    max_length = max(
        max(len(seq) for seq in encoded_inputs), max(len(seq) for seq in encoded_targets)
    )
    print(f"{max_length=}")

    # Pad input and output sequences to have the same max_length length
    # in order to be able to process batches during training
    padded_inputs = [
        seq + [tokenizer.token_to_id["<pad>"]] * (max_length - len(seq)) for seq in encoded_inputs
    ]
    padded_targets = [
        seq + [tokenizer.token_to_id["<pad>"]] * (max_length - len(seq)) for seq in encoded_targets
    ]

    # Prepare a Dataset and DataLoader
    dataset = SequenceDataset(padded_inputs, padded_targets, pad_token_id=tokenizer.token_to_id["<pad>"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    model = EncoderDecoderTransformer(
        vocab_size=len(tokenizer.tokens),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_length=max_length,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model Output Layer Shape: {model.output_layer.weight.shape}")
    print(f"Tokenizer Vocabulary Size: {len(tokenizer.token_to_id)}")

    # Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for src_batch, tgt_batch in dataloader:
            src, tgt_input, tgt_output = (
                src_batch[:, :-1],
                tgt_batch[:, :-1],
                tgt_batch[:, 1:],
            )
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.view(-1, len(tokenizer.tokens)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Generate predictions on a subset of examples
        if True and epoch % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
            test_sequences = [ex[0] for ex in examples]
            predictions = generate_predictions(model, tokenizer, test_sequences, device, max_length=max_length)
            print(f"Predictions after epoch {epoch + 1}:")
            for seq, pred in predictions.items():
                print(f"Input: {seq}, Prediction: {''.join(pred)}")
