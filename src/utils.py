# Small helpers used by all scripts.
import os, re, json, random
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

PAD_IDX = 0   # reserved pad id
OOV_IDX = 1   # out-of-vocab id

def seed_everything(seed: int = 42):
    """Make runs reproducible across Python/NumPy/PyTorch."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# very simple tokenizer: keep alnum tokens only
_word_re = re.compile(r"[a-z0-9]+", re.IGNORECASE)
def simple_tokenize(text: str):
    return _word_re.findall(str(text).lower())

def build_vocab(token_lists: List[List[str]], max_size: int = 10000) -> Dict[str, int]:
    """Top-k vocabulary from training tokens. 0/1 are reserved for PAD/OOV."""
    cnt = Counter()
    for toks in token_lists:
        cnt.update(toks)
    vocab = {"<PAD>": PAD_IDX, "<OOV>": OOV_IDX}
    for i, (tok, _) in enumerate(cnt.most_common(max_size - 2), start=2):
        vocab[tok] = i
    return vocab

def encode(tokens: List[str], vocab: Dict[str,int]):
    """Map tokens → ids using vocab (OOV → 1)."""
    return [vocab.get(t, OOV_IDX) for t in tokens]

@dataclass
class SentimentExample:
    ids: List[int]
    label: int  # 1=positive, 0=negative

class ReviewsDataset(Dataset):
    """We’ll pad/truncate on-the-fly to the seq_len chosen for a run."""
    def __init__(self, examples, seq_len: int):
        self.examples = examples; self.seq_len = seq_len
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = ex.ids[:self.seq_len] + [PAD_IDX] * max(0, self.seq_len - len(ex.ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ex.label, dtype=torch.float32)

def make_loader(examples, seq_len: int, batch_size: int, shuffle: bool):
    return DataLoader(ReviewsDataset(examples, seq_len), batch_size=batch_size, shuffle=shuffle, num_workers=0)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Binary metrics used in the assignment (Accuracy + macro F1)."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)
