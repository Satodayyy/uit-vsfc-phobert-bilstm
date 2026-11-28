from .du_lieu import tap_con
from .thiet_ke import CASES

CONF_RNN = {"lr": .001, "hidden": 256, "layers": 1, "dropout": .3}
CONF_PHO = {"lr": .00002, "dropout": .1}
CONF_BI = {"lr": .0005, "hidden": 256, "layers": 1, "dropout": .4}
CONF_TUNE = [CONF_RNN, CONF_BI, {"lr": .0005, "hidden": 128, "layers": 1, "dropout": .4}]

def mau_doi_chung(data, pct=100):
    pool = tap_con(data["train"], 3, 42)
    return {"train": tap_con(pool, pct, 42), "validation": tap_con(data["validation"], 20, 42)}

def cau_hinh_tung_luot():
    cac_luot = {}
    def them(ma, ho, config, seed=42, pool=False, pct=100, **thay_doi):
        cac_luot[ma] = dict(spec=dict(CASES[ho], **thay_doi), config=config,
                          seed=seed, pool=pool, pct=pct)
    for seed in (42,43,44):
        them(f"bi_{seed}", "bilstm", CONF_BI, seed)
        them(f"att_{seed}", "attention", CONF_RNN, seed)
    them("att_sans_att", "attention_bo_attn", CONF_RNN)
    them("att_sans_ft", "attention_bo_ft", CONF_RNN)
    for pct in (10,25,50,100):
        them(f"petit_bi_{pct}", "bilstm", CONF_RNN, pool=True, pct=pct)
        them(f"petit_pho_{pct}", "phobert", CONF_PHO, pool=True, pct=pct)
    them("petit_bi_syll", "bilstm", CONF_RNN, pool=True, tach=False)
    them("petit_pho_syll", "phobert", CONF_PHO, pool=True, tach=False)
    them("petit_pho_4", "phobert_4_tang", CONF_PHO, pool=True)
    them("petit_pho_gel", "phobert_dong_bang", CONF_PHO, pool=True)
    them("petit_pho_sans_poids", "phobert_bo_trong_so", CONF_PHO, pool=True)
    for pct in (10,25,50):
        them(f"courbe_bi_{pct}", "bilstm", CONF_BI)
        cac_luot[f"courbe_bi_{pct}"]["spec"]["phan_tram"]=pct
    them("att_syll", "attention_am_tiet", CONF_RNN)
    return cac_luot

def huan_luyen(ma, du_lieu, thu_muc):
    from .hoc_sau import chay_mot_luot
    p=cau_hinh_tung_luot()[ma]
    data=mau_doi_chung(du_lieu,p["pct"]) if p["pool"] else du_lieu
    return chay_mot_luot(data,p["spec"],p["config"],p["seed"],thu_muc,
                         epochs_tham_do=3 if p["pool"] else None)
