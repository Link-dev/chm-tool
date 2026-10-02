"""The published weights (folder weights/ of the Hugging Face dataset Link-Dev/canopy-height-data): download and
sha256 check.

  source/UNet-*.pth                 UNet-ALS of every input representation (pre-trained on USGS 3DEP airborne lidar)
  <SITE>/{UNet-SLS,KG-UNet1,KG-UNet2}.pth   the paper's models of the international sites (input AE)
"""
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://huggingface.co/datasets/Link-Dev/canopy-height-data/resolve/main/weights"

SHA256 = {
    "source/UNet-ALS.pth": "b86e3b93e6b25d31326a67da243f098d959067fdbe3cdda731a242370a554a81",
    "source/UNet-A-ALS.pth": "41f0897b704f71581e38159d9716d0d52706bbbedefbc3007e48fad376a0ba1e",
    "source/UNet-E-ALS.pth": "f4796f364f30026a603fc2ac59023cdfc136f1018ae9cb966c454a4b32e7c97a",
    "source/UNet-T-ALS.pth": "61fecd223dc46b8aed38d4df67fe8fddf09a19d188419e0e7138d401f23ba0fa",
    "source/UNet-TE-ALS.pth": "b51ba6fc6861b0dc51feb4c680ec2e1fb27543c6d67dee9344372d774f6f9b86",
    "EBR/UNet-SLS.pth": "fbde2e52fdd067bc845bcf41eccae80965eb7ac0b03ae078c21600e48d6c7398",
    "EBR/KG-UNet1.pth": "7b1e16036fc21c3bfe223edf16d858e9e80eb1d3657c6b707f30cb4781f770d0",
    "EBR/KG-UNet2.pth": "43935effdeca4e6ac7d5eddc1df1322a403a78d164d586b8d959a1c8bad60af8",
    "MRF/UNet-SLS.pth": "a2c952ac9151402a1de6bd8d341924cf059338c49417d6b1a867e09f243fd95a",
    "MRF/KG-UNet1.pth": "6fa8f96c2477fff0f69bd5a6bd171c8259f492a1c7f758dd37f5fd6109490380",
    "MRF/KG-UNet2.pth": "1ba44371cc3ec9e97574401b3dc56e32b8a39e13bfaad9033010a0b7161612f9",
    "MUR/UNet-SLS.pth": "f30a03e36749b922c8f43d1098a073cf2d688e6810e8436e0047d8c32c2ad0fe",
    "MUR/KG-UNet1.pth": "3237b8a4e8350d62a1b449ec26700e734b5b27a6e93f80eb6040435d15f54506",
    "MUR/KG-UNet2.pth": "f54da18e0b9ad3c88ad957fb8660e1bd00a77c9ac3844bba91d016e16852e1d8",
    "SER/UNet-SLS.pth": "76bc433880fc5a19da153cd7ff33f5f545944c9416cf9bed73252873b7bc6669",
    "SER/KG-UNet1.pth": "af65f3f69cc52578518cf12139bd2335966945a4d5be47804c097fca1ee1c5cc",
    "SER/KG-UNet2.pth": "22fc5471b164b4314d231f023a0dc0ed4d8766e49d8349102f63a390ce37c888",
    "SPC/UNet-SLS.pth": "bd587313ee4a76dbc477bf6aa979ea93a36edd46d30c99781501de9b883a0b14",
    "SPC/KG-UNet1.pth": "7c242eb8ff3e218501691ec5164c8f62183a5cbac41e778de5128c755e80bc1f",
    "SPC/KG-UNet2.pth": "c754f7f5c5635a6a538832d824725d60829bdcca46cdf19111754c9365abd7b0",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(relpath, source=WEIGHTS_URL, dest="weights", log=print):
    """dest/relpath, checked against its sha256 (SHA256): kept if already there, else copied from a folder
    (source/relpath or source/<file name>, e.g. the dataset's weights/ folder on Google Drive) or downloaded from a
    URL prefix (source/relpath)."""
    want = SHA256[relpath]
    out = Path(dest) / relpath
    if out.exists() and sha256(out) == want:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    s = str(source)
    if s.startswith(("http://", "https://")):
        url = s.rstrip("/") + "/" + relpath
        log(f"downloading {url}")
        urllib.request.urlretrieve(url, tmp)
    else:
        cands = [Path(s) / relpath, Path(s) / Path(relpath).name]
        src = next((c for c in cands if c.exists()), None)
        if src is None:
            raise FileNotFoundError(f"{relpath} not found in {s} (looked for {', '.join(map(str, cands))})")
        log(f"copying {src}")
        shutil.copyfile(src, tmp)
    got = sha256(tmp)
    if got != want:
        tmp.unlink()
        raise ValueError(f"{relpath}: sha256 {got} differs from the published {want} (incomplete or wrong file)")
    os.replace(tmp, out)
    return out
