from pathlib import Path
from uuid import uuid4
import shutil
from .du_lieu import checksum, doc, ghi

KICH_THUOC=18*1024*1024

def tach_checkpoint(source,dest):
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    ds=[]
    with Path(source).open("rb") as reader:
        while data:=reader.read(KICH_THUOC):
            ten=f"trong_so.{len(ds)+1:03d}"
            (dest/ten).write_bytes(data)
            ds.append({"tep":ten,"bytes":len(data),"sha256":checksum(dest/ten)})
    ghi(dest/"chi_muc.json",{"sha256":checksum(Path(source)),"so_byte":Path(source).stat().st_size,"phan":ds})

def ghep_checkpoint(source,dest):
    source,dest=Path(source),Path(dest)
    data=doc(source/"chi_muc.json")
    dest.mkdir(parents=True,exist_ok=True)
    out=dest/"model.pt"
    if out.is_file() and out.stat().st_size==data["so_byte"] and checksum(out)==data["sha256"]:
        return out
    tmp=dest/f"{uuid4().hex}.tmp"
    try:
        with tmp.open("wb") as writer:
            for item in data["phan"]:
                f=source/item["tep"]
                if not f.is_file() or checksum(f)!=item["sha256"]:
                    raise ValueError(f"Thiếu hoặc sai mã kiểm tra của {f.name}.")
                with f.open("rb") as reader:
                    shutil.copyfileobj(reader,writer)
        if checksum(tmp)!=data["sha256"]:
            raise ValueError("Không khớp SHA-256 sau khi ghép checkpoint.")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out

def thu_muc_suy_luan(goc):
    goc=Path(goc)
    cache=goc/"cache"/"suy_luan_phobert"
    src=goc/"mo_hinh"/"pho_42"
    ghep_checkpoint(src,cache)
    ghi(cache/"run.json",doc(src/"run.json"))
    return cache
