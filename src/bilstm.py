import gzip
import hashlib
from uuid import uuid4
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .du_lieu import CACHE, FT_URL, tai

def tu_dien(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def tai_vector(vocab):
    wanted = set(vocab) - {"<PAD>", "<UNK>"}
    key = hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest()[:24]
    saved = CACHE / f"fasttext-{key}.npz"
    if saved.exists():
        with np.load(saved, allow_pickle=False) as f:
            return dict(zip(f["tu"].tolist(), f["vector"]))
    path = tai(FT_URL, CACHE / "cc.vi.300.vec.gz")
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        _, dim = map(int, next(f).split())
        assert dim == 300
        for line in f:
            word, _, rest = line.partition(" ")
            if word in wanted:
                vector = np.fromstring(rest, dtype="float32", sep=" ")
                if vector.shape != (300,) or not np.isfinite(vector).all():
                    raise ValueError("Vector fastText không hợp lệ.")
                found[word] = vector
    if not found:
        raise ValueError("Không có từ chung với fastText.")
    temp = saved.with_name(saved.stem + f".{uuid4().hex}.npz")
    np.savez_compressed(temp, tu=np.array(list(found)), vector=np.stack(list(found.values())))
    temp.replace(saved)
    return found

