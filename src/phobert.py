from functools import lru_cache
from transformers import AutoConfig, AutoModelForSequenceClassification, PhobertTokenizer
from .du_lieu import CACHE, PHO_REV, tai

@lru_cache(maxsize=1)
def pho_path(trong_so=False):
    root = CACHE / ("phobert-" + PHO_REV)
    url = f"https://huggingface.co/vinai/phobert-base-v2/resolve/{PHO_REV}/"
    files = ["bpe.codes", "vocab.txt", "config.json"]
    if trong_so:
        files.append("pytorch_model.bin")
    for filename in files:
        tai(url + filename, root / filename)
    return root


@lru_cache(maxsize=1)
def tokenizer():
    root = pho_path()
    # Dùng đúng bộ mã hóa BPE của PhoBERT.
    return PhobertTokenizer(vocab_file=str(root / "vocab.txt"), merges_file=str(root / "bpe.codes"))


def tao_phobert(spec, conf, tu_pretrained=True):
    cfg = AutoConfig.from_pretrained(pho_path(), num_labels=3, classifier_dropout=conf["dropout"])
    if tu_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(pho_path(True), config=cfg)
    else:
        model = AutoModelForSequenceClassification.from_config(cfg)
    if spec["mo_tang"] != "tat_ca":
        model.base_model.requires_grad_(False)
        if spec["mo_tang"] == "4_cuoi":
            for layer in model.base_model.encoder.layer[-4:]:
                layer.requires_grad_(True)
    return model
