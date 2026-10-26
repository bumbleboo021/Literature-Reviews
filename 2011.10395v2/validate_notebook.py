"""Execute the notebook in a local kernel, validate it, and extract plots for review."""
import base64
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
for key, folder in {
    "JUPYTER_RUNTIME_DIR": ".cache/jupyter_runtime",
    "IPYTHONDIR": ".cache/ipython",
    "MPLCONFIGDIR": ".cache/matplotlib",
}.items():
    destination = ROOT / folder
    destination.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(destination)

kernel_root = ROOT / ".cache/jupyter"
kernel_dir = kernel_root / "kernels/python3"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Python 3", "language": "python",
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(kernel_root)
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

import nbformat
from nbclient import NotebookClient

path = ROOT / "understanding_differentiable_quantum_circuits.ipynb"
nb = nbformat.read(path, as_version=4)
start = time.perf_counter()
def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}", flush=True)

client = NotebookClient(nb, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=progress)
client.execute()
nbformat.validate(nb)
nbformat.write(nb, path)
plots = ROOT / "tmp/notebook_plots"
plots.mkdir(parents=True, exist_ok=True)
count = 0
for i, cell in enumerate(nb.cells):
    for output in cell.get("outputs", []):
        assert output.output_type != "error", (i, output)
        if output.output_type == "stream":
            print(output.text, end="")
        png = output.get("data", {}).get("image/png")
        if png:
            count += 1
            (plots / f"plot_{count:02d}.png").write_bytes(base64.b64decode(png))
print(f"Validated {len(nb.cells)} cells, {count} plots, {time.perf_counter()-start:.1f} seconds.", flush=True)
