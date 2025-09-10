#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 22:51:47 2025

@author: chenzhijing
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Language Modeling with Recurrent Neural Networks: RNN vs. LSTM
Author: Zhijing
What this script does:
  1) Download & minimally clean the corpus (LM-friendly preprocessing)
  2) Word tokenization; add <sos>/<eos>; build vocab with <pad>/<unk>/<sos>/<eos>
  3) Make 80/10/10 split and a PyTorch Dataset that yields (x, y) next-token windows
  4) Define RNN and LSTM models (Embedding -> RNN/LSTM -> Linear)
  5) Train with Adam/AdamW, dropout, gradient clipping at 1.0
  6) Evaluate Validation/Test Perplexity; generate samples at T in {0.7, 1.0, 1.3}
"""

import os, re, math, random, time, requests
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Tokenization (NLTK) ----
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download("punkt", quiet=True)

# ==========================
#       Hyperparameters
# ==========================
URL = "https://dgoldberg.sdsu.edu/515/harrypotter.txt"

# Model/Training knobs (tweak these for ablations)
MODEL_TYPE = "LSTM"    # "RNN" or "LSTM"
EMBED_SIZE = 256       # 128–256 per spec
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2          # try 0.0 vs 0.2 for ablation
SEQ_LEN = 256          # Context length: try 128 VS 256 for ablation
BATCH_SIZE = 64
GRAD_CLIP = 1.0
STRIDE = None  # None => non-overlap windows; else <= SEQ_LEN
LR = 3e-4      
MAX_EPOCHS = 20

    

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECIALS = {"pad": "<pad>", "unk": "<unk>", "sos": "<sos>", "eos": "<eos>"}

# ==========================
#   1) Download & Clean
# ==========================
def minimal_clean(text: str) -> str:
    """Minimal LM cleaning:
    - collapse whitespace
    - keep only [a-zA-Z0-9 .]
    - lowercase
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\. ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Downloading corpus...")
resp = requests.get(URL, timeout=30)
resp.raise_for_status()  
raw_text = resp.text
print(f"Total characters: {len(raw_text)}")
print("Raw sample:", raw_text[:200].replace("\n", " "))

cleaned = minimal_clean(raw_text)
print("\nCleaned sample:", cleaned[:200])

# ==========================
#   2) Tokenize to words
# ==========================
# Option A (simple): token on full text (treat as one stream)
# Option B (recommended): split to sentences, add <sos>/<eos> per sentence
def tokenize_with_sentence_marks(text: str):
    sents = sent_tokenize(text)
    sents = [s.strip() for s in sents if s.strip()]
    toks = []
    for s in sents:
        words = word_tokenize(s)
        if not words:
            continue
        toks.extend([SPECIALS["sos"], *words, SPECIALS["eos"]])
    return toks

tokens = tokenize_with_sentence_marks(cleaned)
print(f"\nTotal tokens: {len(tokens)}")
print("Tokens sample:", tokens[:30])

# ==========================
#   3) Split first, then build Vocabulary (avoid leakage)
# ==========================
n = len(tokens)
train_tokens = tokens[:int(0.8*n)]
val_tokens   = tokens[int(0.8*n):int(0.9*n)]
test_tokens  = tokens[int(0.9*n):]

# ==========================
#   3b) Build Vocabulary (FROM TRAIN ONLY)
# ==========================
class Vocab:
    def __init__(self, tokens, min_freq=1):
        counter = Counter(tokens)
        self.specials = [SPECIALS["pad"], SPECIALS["unk"], SPECIALS["sos"], SPECIALS["eos"]]
        self.itos = list(self.specials)
        for w, c in counter.most_common():
            if c >= min_freq and w not in self.specials:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[SPECIALS["pad"]]
        self.unk_id = self.stoi[SPECIALS["unk"]]
        self.sos_id = self.stoi[SPECIALS["sos"]]
        self.eos_id = self.stoi[SPECIALS["eos"]]

    def __len__(self): return len(self.itos)
    def encode(self, toks): return [self.stoi.get(t, self.unk_id) for t in toks]
    def decode(self, ids):  return [self.itos[i] for i in ids]

# [修改] 仅用训练tokens建词表，避免验证/测试泄漏
vocab = Vocab(train_tokens, min_freq=1)
print("Vocab size:", len(vocab))

encoded_train = vocab.encode(train_tokens)
encoded_val   = vocab.encode(val_tokens)
encoded_test  = vocab.encode(test_tokens)
print(f"Split sizes -> train:{len(encoded_train)}  val:{len(encoded_val)}  test:{len(encoded_test)}")

# ==========================
#   4) Dataset / DataLoader
# ==========================
class LMWindowDataset(Dataset):
    """Yield (x, y) windows for next-token prediction:
       x: [t0..t_{L-1}], y: [t1..t_L]
    """
    def __init__(self, ids, seq_len=128, stride=None):
        self.ids = ids
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else self.seq_len
        self.starts = []
        i = 0
        while i + self.seq_len < len(self.ids):
            self.starts.append(i)
            i += self.stride

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

train_ds = LMWindowDataset(encoded_train, SEQ_LEN, STRIDE)
val_ds   = LMWindowDataset(encoded_val,   SEQ_LEN, STRIDE)
test_ds  = LMWindowDataset(encoded_test,  SEQ_LEN, STRIDE)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ==========================
#   5) Models: RNN & LSTM
# ==========================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        o, h = self.rnn(x, hidden)
        logits = self.fc(o)
        return logits, h

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        o, h = self.lstm(x, hidden)
        logits = self.fc(o)
        return logits, h

def build_model(kind: str):
    if kind.upper() == "RNN":
        return RNNModel(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    elif kind.upper() == "LSTM":
        return LSTMModel(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    else:
        raise ValueError("MODEL_TYPE must be 'RNN' or 'LSTM'")

model = build_model(MODEL_TYPE).to(DEVICE)
print(f"\nModel: {MODEL_TYPE}\nParameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ==========================
#   6) Training Utilities
# ==========================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits, _ = model(x)
            # Flatten: (B*T, V) vs (B*T)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def generate(model, prefix_tokens, max_new_tokens=50, temperature=1.0):
    """Greedy sampling with temperature; word-level."""
    model.eval()
    ids = vocab.encode(prefix_tokens)
    ids = ids[:SEQ_LEN] if len(ids) > SEQ_LEN else ids
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    hidden = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, hidden = model(x[:, -SEQ_LEN:], hidden)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sampling
            x = torch.cat([x, next_id], dim=1)
    return " ".join(vocab.decode(x[0].tolist()))

# ==========================
#   7) Train Loop (+ EarlyStopping & Scheduler)
# ==========================
class EarlyStopper:
    def __init__(self, patience=4, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, val_loss, model):
        if (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad = 0
            if self.restore_best:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, cooldown=0, min_lr=0.0, eps=1e-8
)
early = EarlyStopper(patience=4, min_delta=1e-4, restore_best=True)

best_val = float("inf")
ckpt_name = f"best_{MODEL_TYPE.lower()}.pt"  
t0 = time.time()
train_losses, val_losses = [], []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss, running_tokens = 0.0, 0

    for x, y in train_dl:
        x = x.to(DEVICE); y = y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        # Gradient clipping per assignment guideline
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item() * y.numel()
        running_tokens += y.numel()

    train_loss = running_loss / running_tokens
    val_loss, val_ppl = evaluate(model, val_dl, criterion)
    train_losses.append(train_loss); val_losses.append(val_loss)

    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} (ppl {math.exp(train_loss):.2f}) "
          f"| val_loss {val_loss:.4f} (ppl {val_ppl:.2f}) | lr {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model_state": model.state_dict(),
                    "vocab": vocab.itos,
                    "config": {
                        "MODEL_TYPE": MODEL_TYPE, "EMBED_SIZE": EMBED_SIZE,
                        "HIDDEN_SIZE": HIDDEN_SIZE, "NUM_LAYERS": NUM_LAYERS,
                        "DROPOUT": DROPOUT, "SEQ_LEN": SEQ_LEN}}, ckpt_name)

    if early.step(val_loss, model):
        print("Early stopping triggered.")
        break

early.restore(model)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/60:.1f} min.")

# ==========================
#   Plot loss curves
# ==========================
import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)
plt.figure()
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"{MODEL_TYPE} Loss Curves (seq_len={SEQ_LEN}, dropout={DROPOUT})")
plt.legend()
plt.grid(True)
plt.tight_layout()

fname = f"loss_curve_{MODEL_TYPE}_seq{SEQ_LEN}_drop{DROPOUT}.png"
plt.savefig(fname, dpi=200)
# plt.show()
print(f"Saved loss curve to {fname}")

# ==========================
#   8) Final Evaluation
# ==========================
ckpt = torch.load(ckpt_name, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
test_loss, test_ppl = evaluate(model, test_dl, criterion)
print(f"\nBest-val checkpoint -> Test loss {test_loss:.4f} | Test PPL {test_ppl:.2f}")

# ==========================
#   9) Generation Samples
# ==========================
prompts = [
    ["<sos>", "harry", "looked", "at", "ron", "and", "said"],
    ["<sos>", "dumbledore", "whispered"],
    ["<sos>", "the", "castle", "was"]
]
for T in (0.7, 1.0, 1.3):
    print(f"\n--- Samples (T={T}) ---")
    for p in prompts:
        text = generate(model, p, max_new_tokens=40, temperature=T)
        print(text)
