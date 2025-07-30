import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class WHUCADDataset(Dataset):
    """
    Loads WHUCAD script files (*.vec or *.txt) containing CADQuery-like
    commands. Expects train/*.txt and test/*.txt under the mesh directory.
    """
    def __init__(self, split="train", data_dir="data/whucad/mesh", vocab=None, max_length=512):
        self.files = sorted(glob.glob(os.path.join(data_dir, split, "*.txt")))
        self.max_length = max_length
        if vocab is None and split=="train":
            self.vocab = self.build_vocab(self.files)
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError("Vocab must be passed for non-train split")
        self.unk = self.vocab.get("<unk>")
        self.bos = self.vocab.get("<bos>")
        self.eos = self.vocab.get("<eos>")

    def build_vocab(self, files, min_freq=2):
        from collections import Counter
        counter = Counter()
        for path in files:
            text = open(path, 'r', encoding='utf8').read()
            tokens = self.tokenize(text)
            counter.update(tokens)
        specials = ["<pad>","<unk>","<bos>","<eos>"]
        tokens = [tok for tok,f in counter.items() if f>=min_freq]
        idx2tok = specials + sorted(tokens)
        return {tok:i for i,tok in enumerate(idx2tok)}

    def tokenize(self, text):
        import re
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)

    def encode(self, tokens):
        seq = [self.bos] + [self.vocab.get(t,self.unk) for t in tokens][:self.max_length-2] + [self.eos]
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        text = open(self.files[idx], 'r', encoding='utf8').read()
        tokens = self.tokenize(text)
        return self.encode(tokens)

def collate_fn(batch):
    seqs = batch
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded[:, :-1], padded[:, 1:]

def get_dataloaders(data_dir="data/whucad/mesh", batch_size=8, max_length=512):
    train_ds = WHUCADDataset("train", data_dir, vocab=None, max_length=max_length)
    test_ds = WHUCADDataset("test",  data_dir, vocab=train_ds.vocab, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader, train_ds.vocab
