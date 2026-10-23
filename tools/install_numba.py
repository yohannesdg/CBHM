"""Install numba on a slow or unstable connection.

pip restarts a download from zero when the connection drops, so the 40 MB llvmlite wheel may never finish.
This script asks PyPI which wheels fit this Python, downloads them with the app's resumable downloader
(HTTP Range, many retries, partial files kept in data/raw/wheels), and then installs from those files.
Run it again after an interruption: it continues where it stopped.
"""
from __future__ import annotations

import json
import platform
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prep import prep_volta as pv  # noqa: E402

WHEELS = ROOT / "data" / "raw" / "wheels"


def platform_tag():
    bits = struct.calcsize("P") * 8
    if sys.platform == "win32":
        return "win_amd64" if bits == 64 else "win32"
    if sys.platform == "darwin":
        return "macosx"
    return {"AMD64": "x86_64"}.get(platform.machine(), platform.machine())


def pick(project, version=None):
    url = f"https://pypi.org/pypi/{project}/json" if version is None else f"https://pypi.org/pypi/{project}/{version}/json"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": pv.UA}), timeout=60) as r:
        info = json.load(r)
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tag = platform_tag()
    files = [f for f in info["urls"] if f["packagetype"] == "bdist_wheel" and f"-{py}-" in f["filename"] and tag in f["filename"]]
    if not files:
        raise SystemExit(f"PyPI has no {project} {info['info']['version']} wheel for {py} {tag}")
    return info, files[0]


def requirement_version(info, name):
    """'llvmlite<0.51,>=0.50.0dev0' -> the newest llvmlite release that satisfies it is found by asking PyPI."""
    for req in info["info"].get("requires_dist") or []:
        if req.split(";")[0].strip().lower().startswith(name):
            return req.split(";")[0].strip()
    return None


def main():
    WHEELS.mkdir(parents=True, exist_ok=True)
    print(f"Python {platform.python_version()} ({platform_tag()}), installing into {sys.executable}")
    try:
        import numba  # noqa: F401
        print(f"numba {numba.__version__} is already installed")
        return 0
    except ImportError:
        pass
    nb_info, nb = pick("numba")
    need = requirement_version(nb_info, "llvmlite")
    print(f"numba {nb_info['info']['version']} needs {need}")
    lv_info, lv = pick("llvmlite")
    lo = need.split(">=")[1].split(",")[0] if ">=" in need else None
    if lo and lv_info["info"]["version"].split(".")[:2] != lo.split(".")[:2]:
        lv_info, lv = pick("llvmlite", lo.replace("dev0", "").rstrip("."))
    for f in (lv, nb):
        dest = WHEELS / f["filename"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"have {dest.name}")
            continue
        print(f"downloading {dest.name} ({f['size'] / 1e6:.1f} MB)")
        pv.download(f["url"], dest, retries=40, timeout=60, resume=True)
        if dest.stat().st_size != f["size"]:
            dest.unlink()
            raise SystemExit(f"{dest.name} has the wrong size; run again")
    print("installing from the downloaded files")
    cmd = [sys.executable, "-m", "pip", "install", "--user", "--no-index", "--find-links", str(WHEELS),
           "--disable-pip-version-check", "numba"]
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit("pip could not install the downloaded wheels (see the messages above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
