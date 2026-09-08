"""Regenerate every figure of the paper from the data in this repository.

Usage
-----
    python make_figures.py [--outdir DIR] [--only NAME ...]

Each figure directory holds one plotting script and the data it reads. This
script runs them in order of appearance in the paper, writes the PDFs into
``figures/`` under the paper's file names, and stops with a non-zero exit code
if any script fails or any expected file is missing. Figure 1 is a TikZ drawing;
it is compiled when ``latexmk`` is on the PATH and skipped with a message
otherwise.

Requirements: Python with numpy, matplotlib and pandas (see ``environment.yml``).
No network access, no environment variables, no files outside this repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (directory, script, paper figure files written into the output directory)
FIGURES = [
    ("fig02_13_zalmoxis_literature_mr", "plot_literature_mr.py",
     ["zalmoxis_literature_mr.pdf", "zalmoxis_literature_profiles.pdf"]),
    ("fig03_calliope_atmodeller", "plot_calliope_atmodeller.py",
     ["calliope_atmodeller.pdf"]),
    ("fig04_chili_validation", "plot_chili_validation.py",
     ["chili_validation.pdf"]),
    ("fig05_phi_sweep_m1", "plot_phi_sweep.py", ["phi_sweep_m1.pdf"]),
    ("fig06_contraction_grid", "plot_contraction_grid.py", ["contraction_grid.pdf"]),
    ("fig07_redox", "plot_redox.py", ["redox.pdf"]),
    ("fig08_volatiles", "plot_volatiles.py", ["volatiles.pdf"]),
    ("fig09_robustness", "plot_robustness.py", ["robustness.pdf"]),
    ("fig10_11_12_zalmoxis_first_principles", "plot_first_principles_validation.py",
     ["zalmoxis_spheres.pdf", "zalmoxis_lane_emden.pdf",
      "zalmoxis_conservation_convergence.pdf"]),
    ("fig14_15_16_aragog_first_principles", "plot_aragog_first_principles.py",
     ["aragog_conduction.pdf", "aragog_conservation.pdf", "aragog_transient.pdf"]),
    ("fig17_18_aragog_spider_parity", "plot_aragog_spider_parity.py",
     ["aragog_spider_const.pdf", "aragog_spider_static.pdf"]),
    ("fig19_calliope_atmodeller_outgassing", "plot_outgassing.py",
     ["calliope_atmodeller_outgassing.pdf"]),
]

TIKZ = ("fig01_proteus_loop", "proteus_loop.tex", "proteus_loop.pdf")


def run_script(subdir, script, outdir):
    """Run one plotting script with ``outdir`` as its output directory."""
    cwd = os.path.join(HERE, subdir)
    if script == "plot_literature_mr.py":
        cmd = [sys.executable, script, outdir]  # positional output directory
    else:
        cmd = [sys.executable, script, "--outdir", outdir]
    print(f"[{subdir}] {' '.join(cmd[1:])}", flush=True)
    res = subprocess.run(cmd, cwd=cwd)
    return res.returncode == 0


def compile_tikz(outdir):
    """Compile the Figure 1 schematic with latexmk if it is available."""
    subdir, tex, pdf = TIKZ
    if shutil.which("latexmk") is None:
        print(f"[{subdir}] latexmk not found; skipping {pdf} (a LaTeX install is needed)")
        return True
    cwd = os.path.join(HERE, subdir)
    print(f"[{subdir}] latexmk -pdf {tex}", flush=True)
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", tex],
                         cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        print(f"[{subdir}] latexmk failed; see {os.path.join(cwd, 'proteus_loop.log')}")
        return False
    shutil.copyfile(os.path.join(cwd, pdf), os.path.join(outdir, pdf))
    subprocess.run(["latexmk", "-c", tex], cwd=cwd,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"),
                    help="where the PDFs are written (default: figures/)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only the figure directories whose name contains any of these strings")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    failures = []
    if args.only is None or any(k in TIKZ[0] for k in args.only):
        if not compile_tikz(outdir):
            failures.append(TIKZ[0])
    for subdir, script, files in FIGURES:
        if args.only is not None and not any(k in subdir for k in args.only):
            continue
        ok = run_script(subdir, script, outdir)
        missing = [f for f in files if not os.path.isfile(os.path.join(outdir, f))]
        if not ok or missing:
            failures.append(f"{subdir}: rc={'0' if ok else 'nonzero'} missing={missing}")

    written = sorted(f for f in os.listdir(outdir) if f.endswith(".pdf"))
    print(f"\n{len(written)} PDF files in {outdir}")
    if failures:
        print("FAILED:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("All figures regenerated.")


if __name__ == "__main__":
    main()
