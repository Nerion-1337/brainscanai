"""CI : notebooks valides (nbformat) et sans sorties committées."""
import sys
from pathlib import Path

import nbformat

bad = 0
for p in sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(p, as_version=4)
    nbformat.validate(nb)
    with_out = [c for c in nb.cells if c.cell_type == "code" and c.get("outputs")]
    if with_out:
        print(f"[KO] {p.name} : {len(with_out)} cellule(s) avec sortie (utiliser nbstripout)")
        bad += 1
    else:
        print(f"[OK] {p.name}")
sys.exit(1 if bad else 0)
