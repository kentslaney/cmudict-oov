import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import re

# Parameters
DICT_FILE = "cmudict/cmudict.dict"
BATCH_SIZE = 512
EMBED_SIZE = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Preprocess data
print("Preprocessing data...")
words = []
pronunciations = []

char_vocab = set()
phone_vocab = set()

with open(DICT_FILE, 'r', encoding='latin-1') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        
        # Remove trailing comments
        line = line.split('#')[0].strip()
        parts = line.split()
        if not parts:
            continue
            
        word = parts[0].lower()
        # Remove variant markings like (2)
        word = re.sub(r'\(\d+\)$', '', word)
        
        phones = parts[1:]
        
        words.append(word)
        pronunciations.append(phones)
        
        for c in word:
            char_vocab.add(c)
        for p in phones:
            phone_vocab.add(p)

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"

char_to_idx = {PAD: 0, SOS: 1, EOS: 2}
idx_to_char = {0: PAD, 1: SOS, 2: EOS}
for i, c in enumerate(sorted(list(char_vocab))):
    char_to_idx[c] = i + 3
    idx_to_char[i + 3] = c

phone_to_idx = {PAD: 0, SOS: 1, EOS: 2}
idx_to_phone = {0: PAD, 1: SOS, 2: EOS}
for i, p in enumerate(sorted(list(phone_vocab))):
    phone_to_idx[p] = i + 3
    idx_to_phone[i + 3] = p

print(f"Vocab sizes: Char {len(char_to_idx)}, Phone {len(phone_to_idx)}")
print(f"Input Char Vocab: {char_to_idx}")
print(f"Output Phone Vocab: {phone_to_idx}")
print(f"Total entries: {len(words)}")

class CMUDataset(Dataset):
    def __init__(self, words, pronunciations):
        self.data = []
        for w, p in zip(words, pronunciations):
            x = [char_to_idx[SOS]] + [char_to_idx[c] for c in w] + [char_to_idx[EOS]]
            y = [phone_to_idx[SOS]] + [phone_to_idx[ph] for ph in p] + [phone_to_idx[EOS]]
            self.data.append((x, y))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    xs, ys = zip(*batch)
    
    max_x_len = max(len(x) for x in xs)
    max_y_len = max(len(y) for y in ys)
    
    padded_xs = [x + [char_to_idx[PAD]] * (max_x_len - len(x)) for x in xs]
    padded_ys = [y + [phone_to_idx[PAD]] * (max_y_len - len(y)) for y in ys]
    
    return torch.tensor(padded_xs), torch.tensor(padded_ys)

dataset = CMUDataset(words, pronunciations)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 2. Model Definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # We will transpose in forward
        )
        
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == char_to_idx[PAD]).transpose(0, 1)
        tgt_padding_mask = (tgt == phone_to_idx[PAD]).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src, tgt shape: (seq_len, batch_size)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

        src_emb = self.pos_encoder(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        outs = self.transformer(
            src_emb, tgt_emb, 
            src_mask, tgt_mask, 
            None, 
            src_padding_mask, tgt_padding_mask, src_padding_mask
        )
        return self.out(outs)

model = Seq2SeqTransformer(
    len(char_to_idx), len(phone_to_idx),
    EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=phone_to_idx[PAD])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training
if __name__ == '__main__':
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (src, tgt) in enumerate(dataloader):
            src = src.transpose(0, 1).to(device) # (seq_len, batch_size)
            tgt = tgt.transpose(0, 1).to(device)
            
            tgt_input = tgt[:-1, :]
            tgt_expected = tgt[1:, :]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            
            loss = criterion(output, tgt_expected)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1}/{EPOCHS} Average Loss: {total_loss/len(dataloader):.4f}")

    print("Saving model checkpoint...")
    torch.save(model.state_dict(), "cmudict_transformer.pth")
    print("Model checkpoint saved to cmudict_transformer.pth")

# 4. Inference
def predict(word):
    model.eval()
    word = word.lower()
    src = [char_to_idx[SOS]] + [char_to_idx.get(c, char_to_idx[PAD]) for c in word] + [char_to_idx[EOS]]
    src = torch.tensor(src).unsqueeze(1).to(device) # (seq_len, 1)
    
    tgt_tokens = [phone_to_idx[SOS]]
    
    for _ in range(50):
        tgt = torch.tensor(tgt_tokens).unsqueeze(1).to(device)
        
        with torch.no_grad():
            output = model(src, tgt)
            
        next_token = output[-1, 0, :].argmax().item()
        tgt_tokens.append(next_token)
        
        if next_token == phone_to_idx[EOS]:
            break
            
    phones = [idx_to_phone[idx] for idx in tgt_tokens[1:-1]]
    return " ".join(phones)

if __name__ == '__main__':
    test_words = ["antigravity", "hello", "world", "xylophone"]
    print("\nInference Results:")
    for w in test_words:
        print(f"{w} -> {predict(w)}")
