import gc
import hashlib
import importlib.metadata as md
import math
import os
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .du_lieu import ROOT, SHA, REV, PHO_REV, chuan_hoa, tach_tu, tap_con, doc, ghi, luc_nay
from .chi_so import diem
from .bilstm import BiLSTM, tu_dien, tai_vector
from .phobert import tokenizer, tao_phobert
from .thiet_ke import CFG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))

def chu_ky_hoc():
    import ast
    import json
    files=("du_lieu","chi_so","thiet_ke","bilstm","phobert","hoc_sau")
    contents=[ast.dump(ast.parse((ROOT/"src"/f"{n}.py").read_text(encoding="utf-8"))) for n in files]
    contents.append(json.dumps(CFG,sort_keys=True))
    return hashlib.sha256("\\n".join(contents).encode()).hexdigest()


def tao_mo_hinh(spec, conf, vocab, tu_pretrained=True):
    if spec["ho"] == "phobert":
        return tao_phobert(spec, conf, tu_pretrained)
    return BiLSTM(vocab, conf, spec["attention"])


def ma_hoa_cau(texts, is_pho, vocab, limit=None):
    if is_pho:
        opts = {"truncation": True, "max_length": limit} if limit else {"truncation": False}
        return tokenizer()(list(texts), verbose=False, **opts)["input_ids"]
    return [[vocab.get(w, 1) for w in text.split()[:limit]] or [1] for text in texts]


def tao_batch(sequences, labels, pad, batch, seed, shuffle=False):
    def assembler(rows):
        tokens, targets = zip(*rows)
        return (
            pad_sequence(tokens, batch_first=True, padding_value=pad),
            torch.tensor([len(x) for x in tokens]),
            torch.tensor(targets, dtype=torch.long),
        )

    rows = [
        (torch.as_tensor(seq, dtype=torch.long), int(label))
        for seq, label in zip(sequences, labels)
    ]
    return DataLoader(
        rows,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=assembler,
        generator=torch.Generator().manual_seed(seed),
    )


def tinh_logits(model, tokens, lengths, is_pho):
    tokens, lengths = tokens.to(DEVICE), lengths.to(DEVICE)
    with torch.autocast(
        device_type=DEVICE.type, dtype=torch.float16, enabled=is_pho and DEVICE.type == "cuda"
    ):
        if is_pho:
            mask = torch.arange(tokens.size(1), device=DEVICE)[None, :] < lengths[:, None]
            return model(input_ids=tokens, attention_mask=mask).logits.float()
        return model(tokens, lengths).float()


@torch.inference_mode()
def do_tren_tap(model, batches, is_pho, weights=None):
    model.eval()
    labels, preds, numer, denom = [], [], 0.0, 0.0
    for tokens, lengths, target in batches:
        scores = tinh_logits(model, tokens, lengths, is_pho)
        moved = target.to(DEVICE)
        numer += nn.functional.cross_entropy(scores, moved, weight=weights, reduction="sum").item()
        denom += weights[moved].sum().item() if weights is not None else len(target)
        labels.extend(target.tolist())
        preds.extend(scores.argmax(-1).cpu().tolist())
    return preds, dict(diem(labels, preds), mat_mat=numer / denom)


def chuan_bi_luot(data, spec, config, seed, kiem_tra_ngan=False):
    train = tap_con(data["train"], spec["phan_tram"], seed)
    val = data["validation"]
    field = "tach_tu" if spec["tach"] else "van_ban"
    txt = train[field].tolist()
    pho = spec["ho"] == "phobert"
    vocab = {} if pho else tu_dien(txt)
    long = [len(x) for x in ma_hoa_cau(txt, pho, vocab)]
    p99 = math.ceil(float(np.quantile(long, CFG["token"]["p"])))
    limit = int(np.clip(p99, CFG["token"]["min"], CFG["token"]["max"]))
    batch = CFG["phobert" if pho else "bilstm"]["batch"]
    pad = tokenizer().pad_token_id if pho else 0
    train_y = train.sentiment.to_numpy()
    counts = np.bincount(train_y, minlength=3)
    assert (counts > 0).all()
    weights = None
    if spec["trong_so"]:
        weights = torch.tensor(len(train) / (3 * counts), device=DEVICE, dtype=torch.float)
    train_seq = ma_hoa_cau(txt, pho, vocab, limit)
    train_dl = tao_batch(train_seq, train_y, pad, batch, seed, True)
    val_dl = tao_batch(ma_hoa_cau(val[field], pho, vocab, limit), val.sentiment, pad, batch, seed)
    infos = {
        "vocab": vocab,
        "p99": p99,
        "limit": limit,
        "cot_van_ban": field,
        "pad": pad,
        "spec": spec,
        "config": config,
        "seed": seed,
        "so_cau_train": len(train),
        "so_cau_validation": len(val),
        "ids_train_sha": hashlib.sha256(train.id_dong.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "phien_ban_phobert": PHO_REV,
        "phien_ban_du_lieu": REV,
        "sha_parquet": SHA,
        "thiet_bi": torch.cuda.get_device_name() if DEVICE.type == "cuda" else "CPU",
        "versions": {k: md.version(k) for k in ["underthesea", "torch", "transformers", "numpy"]},
    }
    return train_dl, val_dl, weights, infos


def chay_mot_luot(data, spec, config, seed, folder, *, kiem_tra_ngan=False, epochs_tham_do=None):
    folder.mkdir(parents=True, exist_ok=True)
    info_file = folder / "run.json"
    if info_file.exists() and (folder / "model.pt").exists():
        r = doc(info_file)
        if (r["config"], r["spec"], r["seed"], r["kiem_tra_ngan"], r["phien_ban_du_lieu"]) != (
            config,
            spec,
            seed,
            kiem_tra_ngan,
            REV,
        ):
            raise ValueError(f"Cấu hình đã đổi tại {folder}. Chọn UIT_KET_QUA mới.")
        if r.get("sha_huan_luyen") != chu_ky_hoc():
            raise ValueError("Mã học hoặc cấu hình đã thay đổi. Giữ kết quả cũ và chọn MA_DOT mới.")
        if r.get("epochs_tham_do") != epochs_tham_do:
            raise ValueError("Ngân sách epoch khác lượt đã lưu.")
        return r
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pho = spec["ho"] == "phobert"
    train, valid, weights, info = chuan_bi_luot(data, spec, config, seed, kiem_tra_ngan)
    model = tao_mo_hinh(spec, config, info["vocab"])
    if spec["fasttext"]:
        ft = tai_vector(info["vocab"])
        with torch.no_grad():
            for w, vec in ft.items():
                model.embedding.weight[info["vocab"][w]].copy_(torch.from_numpy(vec))
        info["do_phu_fasttext"] = len(ft) / (len(info["vocab"]) - 2)
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    # Chia batch PhoBERT khi RAM hạn chế, giữ batch hiệu dụng của đề cương.
    micro = min(
        CFG["phobert"]["batch"],
        int(os.getenv("UIT_MICRO_BATCH", "4" if DEVICE.type == "cpu" else "32")),
    )
    if micro < 1:
        raise ValueError("UIT_MICRO_BATCH cần lớn hơn 0.")
    info["sha_huan_luyen"] = chu_ky_hoc()
    info["epochs_tham_do"] = epochs_tham_do
    if pho:
        info["micro_batch"] = micro
    conf = CFG["phobert" if pho else "bilstm"]
    epochs = 1 if kiem_tra_ngan else (epochs_tham_do or conf["epochs"])
    if pho:
        optim = torch.optim.AdamW(
            params, lr=config["lr"], weight_decay=conf["weight_decay"], foreach=False
        )
        steps = epochs * len(train)
        schedule = get_linear_schedule_with_warmup(optim, round(conf["warmup"] * steps), steps)
    else:
        optim = torch.optim.Adam(params, lr=config["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=pho and DEVICE.type == "cuda")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    history, best, best_epoch = [], -1.0, 0
    da_qua, gpu_truoc = 0.0, 0.0
    resume = folder / "tiep_tuc.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        signature = [spec, config, seed, info["ids_train_sha"], epochs, info["sha_huan_luyen"]]
        if state["dau_vet"] != signature:
            raise ValueError("Lượt đang dở dùng cấu hình khác. Kiểm tra thư mục kết quả.")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        if pho:
            schedule.load_state_dict(state["schedule"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        da_qua, gpu_truoc = state["da_qua"], state["gpu_truoc"]
        train.generator.set_state(state["shuffle"])
        torch.set_rng_state(state["rng_cpu"])
        if DEVICE.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state(state["rng_cuda"])
        del state
        gc.collect()
        print(f"Tiếp tục sau epoch {len(history)}.", flush=True)
    start = time.perf_counter()
    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        ys, ps, sum_loss, denom = [], [], 0.0, 0.0
        for tokens, lengths, target in train:
            optim.zero_grad(set_to_none=True)
            total = float(weights[target.to(DEVICE)].sum()) if weights is not None else len(target)
            step = micro if pho else len(target)
            for begin in range(0, len(target), step):
                sub = slice(begin, begin + step)
                part_length = lengths[sub]
                # Giới hạn padding riêng cho micro-batch để giảm bộ nhớ activation.
                maxlen = int(part_length.max())
                scores = tinh_logits(model, tokens[sub, :maxlen], part_length, pho)
                moved = target[sub].to(DEVICE)
                loss_sum = nn.functional.cross_entropy(
                    scores, moved, weight=weights, reduction="sum"
                )
                if pho:
                    loss = loss_sum / total
                else:
                    loss = nn.functional.cross_entropy(scores, moved, weight=weights)
                scaler.scale(loss).backward()
                sum_loss += loss_sum.item()
                ys.extend(target[sub].tolist())
                ps.extend(scores.detach().argmax(-1).cpu().tolist())
            denom += total
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(params, 1.0)
            before = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            if pho and scaler.get_scale() >= before:
                schedule.step()
        _, val = do_tren_tap(model, valid, pho, weights)
        history.append(
            dict(
                epoch=epoch,
                train_f1=diem(ys, ps)["macro_f1"],
                train_loss=sum_loss / denom,
                val_f1=val["macro_f1"],
                val_loss=val["mat_mat"],
                thiet_bi=info["thiet_bi"],
                micro_batch=micro if pho else conf["batch"],
            )
        )
        print(
            f"{folder.name} · epoch {epoch} · Macro-F1 validation {val['macro_f1']:.4f}", flush=True
        )
        eligible = (not pho) or kiem_tra_ngan or epoch >= conf["epoch_chon_tu"]
        if epochs_tham_do is not None:
            eligible = epoch == epochs_tham_do
        if eligible and val["macro_f1"] > best:
            best, best_epoch = val["macro_f1"], epoch
            tmp = folder / "model.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(folder / "model.pt")
        if not pho and epochs_tham_do is None and epoch - best_epoch >= conf["patience"]:
            break
        if epoch < epochs and not kiem_tra_ngan and epochs_tham_do is None:
            # Khôi phục theo epoch cho lượt học dài. Tệp ghi dở không thay bản đã hoàn tất.
            tmp = folder / "tiep_tuc.tmp"
            snap = dict(
                dau_vet=[spec, config, seed, info["ids_train_sha"], epochs, info["sha_huan_luyen"]],
                model=model.state_dict(),
                optim=optim.state_dict(),
                schedule=schedule.state_dict() if pho else {},
                scaler=scaler.state_dict(),
                history=history,
                best=best,
                best_epoch=best_epoch,
                da_qua=da_qua + time.perf_counter() - start,
                gpu_truoc=max(gpu_truoc, torch.cuda.max_memory_allocated() / 2**20)
                if DEVICE.type == "cuda"
                else 0.0,
                shuffle=train.generator.get_state(),
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
            )
            torch.save(snap, tmp)
            tmp.replace(resume)
            del snap
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    info.update(
        giay=da_qua + time.perf_counter() - start,
        tham_so=sum(p.numel() for p in model.parameters()),
        tham_so_hoc=sum(p.numel() for p in params),
        gpu_mib=max(gpu_truoc, torch.cuda.max_memory_allocated() / 2**20)
        if DEVICE.type == "cuda"
        else None,
        validation=best,
        best_epoch=best_epoch,
        history=history,
        kiem_tra_ngan=kiem_tra_ngan,
        ngay=luc_nay(),
    )
    info["thiet_bi"] = " / ".join(dict.fromkeys(e["thiet_bi"] for e in history))
    ghi(info_file, info)
    resume.unlink(missing_ok=True)
    return info


def nap_mo_hinh(folder):
    info = doc(folder / "run.json")
    model = tao_mo_hinh(info["spec"], info["config"], info["vocab"], tu_pretrained=False)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval(), info


@torch.inference_mode()
def xac_suat(model, info, phrase):
    text = chuan_hoa(phrase)
    if not text:
        raise ValueError("Vui lòng nhập một câu phản hồi.")
    if info["spec"]["tach"]:
        text = tach_tu(text)
    pho = info["spec"]["ho"] == "phobert"
    seq = ma_hoa_cau([text], pho, info["vocab"], info["limit"])[0]
    scores = tinh_logits(model, torch.tensor([seq]), torch.tensor([len(seq)]), pho)
    return scores.softmax(-1)[0].cpu().tolist()
