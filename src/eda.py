import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .du_lieu import NHAN


def xuat_eda(data, stats, dest):
    dest.mkdir(parents=True, exist_ok=True)
    summary=pd.DataFrame.from_dict(stats, orient="index").reindex(["train","validation","test"])
    summary.to_csv(dest/"theo_tap.csv", index_label="tap")
    print(summary[["ban_dau","chu_so","cau_co_ma","chu_so_khac","loai_trung","con_lai"]].to_string())
    tr=data["train"]
    lengths=tr.van_ban.str.split().str.len()
    freq=lengths.value_counts().sort_index()
    freq.rename("so_cau").to_csv(dest/"do_dai_train.csv",index_label="so_am_tiet_va_dau_cau")
    fig,ax=plt.subplots(1,2,figsize=(9.4,3.6),layout="constrained")
    counts=tr.sentiment.value_counts().sort_index()
    bars=ax[0].bar(NHAN,counts,color=["#304c61","#cc764e","#27776e"])
    ax[0].bar_label(bars,padding=3)
    ax[0].set(ylim=(0,max(counts)*1.16),ylabel="Số câu",title="Cảm xúc sau lọc trùng")
    ax[1].hist(lengths,bins=range(0,int(lengths.max())+5,4),color="#27776e")
    ax[1].axvline(lengths.quantile(.99),ls="--",color="#974629",label="Phân vị 99%")
    ax[1].legend(frameon=False)
    ax[1].set(xlabel="Số đơn vị cách nhau bằng khoảng trắng",ylabel="Số câu",title="Độ dài câu train")
    for a in ax:a.spines[["right","top"]].set_visible(False)
    fig.savefig(dest/"phan_bo_train.png",dpi=170)
    plt.close(fig)