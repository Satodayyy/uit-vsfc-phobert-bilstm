from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from uuid import uuid4
import re
import shutil
import unicodedata
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from underthesea import word_tokenize

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.getenv("UIT_KET_QUA", ROOT / "ket_qua")).resolve()
CACHE = Path(os.getenv("UIT_CACHE", ROOT / "cache")).resolve()
SEEDS = (42, 43, 44)
SPLITS = ("train", "validation", "test")
IDS = [0, 1, 2]
NHAN = ["tiêu cực", "trung tính", "tích cực"]
REV = "2d76906157232f87e9883fa37ebfc91ffdee1f83"
URL = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/"
PHO_REV = "86cd7fd4c148980922ac11a2cf5e257f2ba639e1"
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz"
SHA = {
    "train": "7545fbab6bcebf2ee8bc529d455edb6ce626f71354c24e0422e9f2832806311a",
    "validation": "68b0a7c97ced26c8c971e0a996719a345e7897870e8d3ea0d319cdce16b90514",
    "test": "74a223204066690a2d6b6c3dac85178e8acc43b0a7f2b11e25e52d5837911e78",
}
MA = re.compile(r"\bwzjwz[0-9]+\b")
CHU_SO = re.compile(r"\b\w*[a-zA-Z]\d\w*\b|\b\w*\d[a-zA-Z]\w*\b")
CHU_LAP = re.compile(r"([^\W\d_])\1{2,}")


def luc_nay():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")


def ghi(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def doc(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tai(url, path, expected=None):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".{uuid4().hex}.part")
        try:
            with urlopen(url, timeout=120) as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            if expected and checksum(temp) != expected:
                raise ValueError(f"Sai mã kiểm tra khi tải {path.name}.")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    if expected and checksum(path) != expected:
        raise ValueError(f"Nội dung {path} khác bản dữ liệu đã chốt.")
    return path


def chuan_hoa(text):
    value = unicodedata.normalize("NFC", text).lower()
    return " ".join(CHU_LAP.sub(r"\1\1", MA.sub("mawzjwz", value)).split())


def tach_tu(text):
    return " ".join(word_tokenize(text, format="text").split()) or text


def khoa(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def dau_vet_tien_xu_ly():
    import ast
    import inspect
    funcs=(chuan_hoa,tach_tu,khoa,du_lieu)
    sources=[ast.dump(ast.parse(inspect.getsource(f))) for f in funcs]
    rule="|".join([REV,*SHA.values(),MA.pattern,CHU_SO.pattern,CHU_LAP.pattern,*sources])
    return hashlib.sha256(rule.encode()).hexdigest()[:16]


def du_lieu():
    
    import importlib.metadata as md

    cach = f"e0-{dau_vet_tien_xu_ly()}-{md.version('underthesea')}-{md.version('underthesea_core')}"
    out = CACHE / cach
    if (out / "thong_ke.json").is_file() and all((out / f"{s}.parquet").exists() for s in SPLITS):
        return {s: pd.read_parquet(out / f"{s}.parquet") for s in SPLITS}, doc(
            out / "thong_ke.json"
        )
    out.mkdir(parents=True, exist_ok=True)
    df, tables, seen = {}, {}, [set(), set(), set()]
    for split in reversed(SPLITS):
        p = tai(
            f"{URL}{REV}/default/{split}/0000.parquet",
            CACHE / "goc" / f"{split}.parquet",
            SHA[split],
        )
        names = json.loads(pq.read_schema(p).metadata[b"huggingface"])["info"]["features"][
            "sentiment"
        ]["names"]
        assert names == ["negative", "neutral", "positive"]
        f = pd.read_parquet(p, columns=["sentence", "sentiment"])
        assert f.sentence.map(lambda text: isinstance(text, str) and bool(text.strip())).all()
        assert f.sentiment.isin(IDS).all()
        f["id_dong"] = np.arange(len(f))
        lower = f.sentence.str.normalize("NFC").str.lower()
        f["van_ban"] = f.sentence.map(chuan_hoa)
        f["tach_tu"] = f.van_ban.map(tach_tu)
        keys = [f[name].map(khoa) for name in ["sentence", "van_ban", "tach_tu"]]
        remove = np.logical_or.reduce([(k != "") & k.isin(old) for k, old in zip(keys, seen)])
        clean = f.loc[~remove].reset_index(drop=True)
        for k, old in zip(keys, seen):
            assert not ((set(k[~remove]) - {""}) & old)
            old.update(set(k) - {""})
        counts = lower.str.count(MA)
        do_dai = f.sentence.str.split().str.len()
        tables[split] = {
            "ban_dau": len(f),
            "chu_so": int(lower.str.contains(CHU_SO).sum()),
            "cau_co_ma": int((counts > 0).sum()),
            "luot_ma": int(counts.sum()),
            "chu_so_khac": int(
                lower.str.replace(MA, "mawzjwz", regex=True).str.contains(CHU_SO).sum()
            ),
            "loai_trung": int(remove.sum()),
            "con_lai": len(clean),
            "phan_bo_goc": [int((f.sentiment == c).sum()) for c in IDS],
            "phan_bo_sau_loc": [int((clean.sentiment == c).sum()) for c in IDS],
            "do_dai_trung_vi": float(do_dai.median()),
            "do_dai_p99": float(do_dai.quantile(0.99)),
        }
        df[split] = clean
        tmp = out / f"{split}.{uuid4().hex}.parquet"
        clean.to_parquet(tmp, index=False)
        tmp.replace(out / f"{split}.parquet")
    assert df["test"].id_dong.tolist() == list(range(3166))
    ghi(out / "thong_ke.json", tables)
    return df, tables


def tap_con(frame, pct, seed):
    if pct == 100:
        return frame
    
    rng = np.random.default_rng(seed)
    positions = []
    for label in IDS:
        pool = np.flatnonzero(frame.sentiment.to_numpy() == label)
        chosen = rng.permutation(pool)[: max(1, round(len(pool) * pct / 100))]
        positions.extend(chosen)
    return frame.iloc[sorted(positions)].reset_index(drop=True)
