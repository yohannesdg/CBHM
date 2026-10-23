"""Report which packages the model can import, and where they come from.

Prints to the console and appends to setup_log.txt (next to setup_and_run.bat).
Exit code 0 if the core packages work, 1 otherwise.
"""
import importlib
import sys
import traceback
from pathlib import Path

CORE = ["numpy", "scipy", "pydantic", "starlette", "fastapi", "uvicorn"]
OPTIONAL = ["numba", "xarray", "netCDF4", "rasterio"]
LOG = Path(__file__).resolve().parents[1] / "setup_log.txt"

lines = [f"python {sys.version.split()[0]} at {sys.executable}"]
ok = True
for name in CORE + OPTIONAL:
    try:
        m = importlib.import_module(name)
        where = Path(getattr(m, "__file__", "") or "").parent
        lines.append(f"  OK       {name:<10} {getattr(m, '__version__', ''):<10} {where}")
    except Exception as e:  # noqa: BLE001
        tag = "MISSING  " if name in CORE else "optional "
        if name in CORE:
            ok = False
        lines.append(f"  {tag}{name:<10} {type(e).__name__}: {e}")
        if name in CORE and not isinstance(e, ModuleNotFoundError):
            lines.append("          " + traceback.format_exc().strip().splitlines()[-1])
text = "\n".join(lines)
print(text)
with open(LOG, "a", encoding="utf-8") as f:
    f.write("\n--- package check ---\n" + text + "\n")
sys.exit(0 if ok else 1)
