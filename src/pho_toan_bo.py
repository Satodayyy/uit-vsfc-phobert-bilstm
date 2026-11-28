from pathlib import Path
import math
import time
import random
import os
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from .du_lieu import ghi, luc_nay, tap_con
from .hoc_sau import ma_hoa_cau
from .chi_so import diem
from .phobert import tao_phobert
from .giao_thuc import CONF_PHO
from .thiet_ke import CASES

class BangTuTrain(nn.Module):
    """AdamW chỉ cần hai moment cho các hàng nhận gradient trên train.

    Những hàng chưa gặp có gradient bằng 0 từ đầu, nên chỉ chịu weight decay.
    Khôi phục bảng đủ 64.001 hàng trước khi phát hành checkpoint.
    """
    def __init__(self, source, ids):
        super().__init__()
        self.register_buffer("goc", source.weight.detach())
        rows=torch.tensor(sorted(set(ids)),device=source.weight.device)
        inverse=torch.full((source.num_embeddings,),-1,dtype=torch.long,device=rows.device)
        inverse[rows]=torch.arange(len(rows),device=rows.device)
        self.register_buffer("rows", rows)
        self.register_buffer("inverse", inverse)
        self.weight=nn.Parameter(source.weight.detach()[rows].clone())
        self.decay=1.0

    def forward(self, tokens):
        ix=self.inverse[tokens]
        return torch.where((ix>=0).unsqueeze(-1), self.weight[ix.clamp_min(0)], self.goc[tokens]*self.decay)

    def khoi_phuc_bang(self):
        value=self.goc*self.decay
        value[self.rows]=self.weight.detach()
        return value

def chia_lo(seq, batch, seed=None):
    indices=np.arange(len(seq))
    if seed is None:
        indices=np.array(sorted(indices,key=lambda i:len(seq[i])))
        return [indices[i:i+batch] for i in range(0,len(indices),batch)]
    rng=np.random.default_rng(seed)
    rng.shuffle(indices)
    result=[]
    for q in range(0,len(indices),batch*20):
        part=sorted(indices[q:q+batch*20],key=lambda i:len(seq[i]))
        result.extend(part[i:i+batch] for i in range(0,len(part),batch))
    rng.shuffle(result)
    return result

def tao_lo(seq, ids, device):
    tokens=pad_sequence([torch.tensor(seq[i]) for i in ids],batch_first=True,padding_value=1).to(device)
    return tokens,tokens.ne(1)

def predict_seq(model, seq, batch, device, bf=False, labels=None, weights=None):
    pred=np.empty(len(seq),dtype=np.int64)
    loss=0.
    model.eval()
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf):
        for ids in chia_lo(seq,batch):
            tok,mask=tao_lo(seq,ids,device)
            logits=model(input_ids=tok,attention_mask=mask).logits.float()
            pred[ids]=logits.argmax(-1).cpu().numpy()
            if labels is not None:
                loss+=nn.functional.cross_entropy(logits,labels[ids],weight=weights,reduction="sum").item()
    return pred, (loss/weights[labels].sum().item() if labels is not None else None)

def huan_luyen(data,dest,pct=100):
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    seed=42
    epochs=3
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf=(torch.cuda.is_bf16_supported() if device.type=="cuda" else os.getenv("UIT_BF16","1")=="1")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train,valid=tap_con(data["train"],pct,42),data["validation"]
    allseq=ma_hoa_cau(train.tach_tu,True,{})
    limit=int(np.clip(math.ceil(np.quantile([len(x) for x in allseq],.99)),64,96))
    seq=ma_hoa_cau(train.tach_tu,True,{},limit)
    chuoi_valid=ma_hoa_cau(valid.tach_tu,True,{},limit)
    y=torch.tensor(train.sentiment.to_numpy(),device=device)
    nhan_valid=torch.tensor(valid.sentiment.to_numpy(),device=device)
    w=len(y)/(3*torch.bincount(y).float())
    model=tao_phobert(CASES["phobert"],CONF_PHO).to(device)
    so_tham_so=sum(p.numel() for p in model.parameters())
    table=BangTuTrain(model.roberta.embeddings.word_embeddings,[k for row in seq for k in row]).to(device)
    model.roberta.embeddings.word_embeddings=table
    optim=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01,fused=True)
    steps=epochs*math.ceil(len(seq)/32)
    sched=get_linear_schedule_with_warmup(optim,round(.1*steps),steps)
    info=dict(config=CONF_PHO,spec=dict(CASES["phobert"],phan_tram=pct),seed=seed,so_cau_train=len(train),so_cau_validation=len(valid),
              ids_train_sha=hashlib.sha256(train.id_dong.to_numpy(dtype="<i8").tobytes()).hexdigest(),
              limit=limit,cot_van_ban="tach_tu",vocab={},pad=1,tham_so=so_tham_so,tham_so_hoc=so_tham_so,
              so_hang_cap_nhat=len(table.rows),thiet_bi=str(device),bf16=bf,epochs=epochs,
              batch=32,dropout_encoder=.1,optimizer="AdamW",bat_dau=luc_nay(),history=[])
    begin=time.perf_counter()
    best=-1
    for ep in range(1,epochs+1):
        model.train()
        preds=np.empty(len(seq),dtype=np.int64)
        loss_sum=0.
        for j,ids in enumerate(chia_lo(seq,32,seed+ep)):
            tok,mask=tao_lo(seq,ids,device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf):
                logits=model(input_ids=tok,attention_mask=mask).logits.float()
            loss=nn.functional.cross_entropy(logits,y[ids],weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            lr=optim.param_groups[0]["lr"]
            optim.step();sched.step()
            table.decay*=1-lr*.01
            preds[ids]=logits.detach().argmax(-1).cpu().numpy()
            loss_sum+=loss.item()*w[y[ids]].sum().item()
            if j%50==0: print(f"PhoBERT · epoch {ep} · batch {j+1}/{math.ceil(len(seq)/32)} · {time.perf_counter()-begin:.0f} s",flush=True)
        vp,vl=predict_seq(model,chuoi_valid,32,device,bf,nhan_valid,w)
        val=diem(valid.sentiment,vp)["macro_f1"]
        info["history"].append(dict(epoch=ep,train_f1=diem(train.sentiment,preds)["macro_f1"],
                                    val_f1=val,train_loss=loss_sum/w[y].sum().item(),val_loss=vl))
        if val>best:
            best=val
            standard={k:v.detach().to(device="cpu",dtype=torch.bfloat16) for k,v in model.state_dict().items() if "word_embeddings." not in k}
            standard["roberta.embeddings.word_embeddings.weight"]=table.khoi_phuc_bang().detach().to(device="cpu",dtype=torch.bfloat16)
            torch.save(standard,dest/"model.tmp")
            (dest/"model.tmp").replace(dest/"model.pt")
            info["best_epoch"]=ep
            info["validation"]=val
            del standard
        info["giay"]=time.perf_counter()-begin
        info["ngay"]=luc_nay()
        ghi(dest/"run.json",info)
        print(f"PhoBERT · epoch {ep} · Macro-F1 validation {val:.4f}",flush=True)
    return info
