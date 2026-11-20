from itertools import product
import random
from .du_lieu import ROOT, doc
CFG = doc(ROOT / "configs/e1_e5.json")

BASE = {
    "tach": True,
    "attention": False,
    "fasttext": False,
    "mo_tang": "tat_ca",
    "trong_so": True,
    "phan_tram": 100,
}
FAMILIES = ("bilstm", "attention", "phobert")


def phuong_an():
    base = dict(BASE, ho="bilstm")
    strong = dict(BASE, ho="attention", attention=True, fasttext=True)
    pho = dict(BASE, ho="phobert")
    cases = {
        "bilstm": base,
        "attention": strong,
        "phobert": pho,
        "bilstm_am_tiet": dict(base, tach=False),
        "phobert_am_tiet": dict(pho, tach=False),
        "attention_bo_attn": dict(strong, attention=False),
        "attention_bo_ft": dict(strong, fasttext=False),
        "attention_am_tiet": dict(strong, tach=False),
        "phobert_dong_bang": dict(pho, mo_tang="dong_bang"),
        "phobert_4_tang": dict(pho, mo_tang="4_cuoi"),
        "phobert_bo_trong_so": dict(pho, trong_so=False),
    }
    for name, variant in [("bilstm", base), ("phobert", pho)]:
        for pct in CFG["e4_phan_tram"]:
            if pct < 100:
                cases[f"{name}_{pct}"] = dict(variant, phan_tram=pct)
    return cases


CASES = phuong_an()
GROUPS = {
    "E1": list(FAMILIES),
    "E2": [
        "bilstm_am_tiet",
        "phobert_am_tiet",
        "attention_am_tiet",
        "attention_bo_attn",
        "attention_bo_ft",
    ],
    "E3": ["phobert_dong_bang", "phobert_4_tang", "phobert_bo_trong_so"],
    "E4": [
        f"{family}_{pct}"
        for family in ("bilstm", "phobert")
        for pct in CFG["e4_phan_tram"]
        if pct < 100
    ],
}


def luoi_cau_hinh(family):
    if family == "phobert":
        c = CFG["phobert"]
        return [dict(lr=lr, dropout=dp) for lr, dp in product(c["learning_rate"], c["dropout"])]
    c = CFG["bilstm"]
    grid = [
        dict(lr=lr, hidden=h, layers=n, dropout=d)
        for lr, h, n, d in product(c["learning_rate"], c["hidden"], c["layers"], c["dropout"])
    ]
    first = dict(lr=0.001, hidden=256, layers=1, dropout=0.3)
    grid.remove(first)
    return [first, *random.Random(42).sample(grid, c["so_cau_hinh"] - 1)]
