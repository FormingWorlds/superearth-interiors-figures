"""Regenerate every figure of the paper from the data in this repository.

Usage
-----
    python make_figures.py [--outdir DIR] [--only NAME ...]

Each figure directory holds one plotting script and the data it reads. This
script runs them in order of appearance in the paper, writes the PDFs into
``figures/`` under the paper's file names, and exits non-zero if any script
fails or any expected file is missing. Expected output files are deleted before
their script runs, so a stale file never passes for a fresh one. Figure 1 is a
TikZ drawing: it is compiled when ``latexmk`` is on the PATH; otherwise the
committed ``figures/proteus_loop.pdf`` is copied into the output directory.

``--only`` selects figure directories by substring of their name, for example
``--only redox volatiles`` or ``--only fig06``; a selector that matches nothing
is an error.

Requirements: Python with numpy, matplotlib and pandas (see ``requirements.txt``).
No network access, no environment variables, no files outside this repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMITTED = os.path.join(HERE, "figures")

# (directory, script, paper figure files written into the output directory)
FIGURES = [
    ("fig01_proteus_loop", "proteus_loop.tex", ["proteus_loop.pdf"]),
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


def run_script(subdir, script, outdir):
    """Run one plotting script with ``outdir`` as its output directory."""
    cwd = os.path.join(HERE, subdir)
    cmd = [sys.executable, script, "--outdir", outdir]
    print(f"[{subdir}] {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode == 0


def copy_if_different(src, dst):
    """Copy ``src`` to ``dst`` unless both names are the same file."""
    if os.path.exists(dst) and os.path.samefile(src, dst):
        return
    shutil.copyfile(src, dst)


def compile_tikz(subdir, tex, pdf, outdir):
    """Compile the TikZ schematic with latexmk, or copy the committed PDF."""
    cwd = os.path.join(HERE, subdir)
    target = os.path.join(outdir, pdf)
    if shutil.which("latexmk") is None:
        src = os.path.join(COMMITTED, pdf)
        print(f"[{subdir}] latexmk not found; copying the committed {pdf}")
        if not os.path.isfile(src):
            print(f"[{subdir}] {src} is missing")
            return False
        copy_if_different(src, target)
        return True
    print(f"[{subdir}] latexmk -pdf {tex}", flush=True)
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", tex],
                         cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[{subdir}] latexmk failed:\n{res.stdout[-3000:]}{res.stderr[-1000:]}")
        return False
    copy_if_different(os.path.join(cwd, pdf), target)
    subprocess.run(["latexmk", "-c", tex], cwd=cwd, capture_output=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=COMMITTED,
                    help="where the PDFs are written (default: figures/)")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="run only the figure directories whose name contains one of these strings")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    selected = [entry for entry in FIGURES
                if args.only is None or any(k in entry[0] for k in args.only)]
    for k in args.only or []:
        if not any(k in entry[0] for entry in FIGURES):
            sys.exit(f"--only {k} matches no figure directory")

    failures = []
    for subdir, script, files in selected:
        if script.endswith(".tex"):
            ok = compile_tikz(subdir, script, files[0], outdir)
        else:
            for f in files:  # a stale output must not pass for a fresh one
                target = os.path.join(outdir, f)
                if os.path.isfile(target):
                    os.remove(target)
            ok = run_script(subdir, script, outdir)
        missing = [f for f in files if not os.path.isfile(os.path.join(outdir, f))]
        if not ok or missing:
            failures.append(f"{subdir}: rc={'0' if ok else 'nonzero'} missing={missing}")

    expected = [f for _, _, files in selected for f in files]
    present = [f for f in expected if os.path.isfile(os.path.join(outdir, f))]
    print(f"\n{len(present)} of {len(expected)} expected PDF files present in {outdir}")
    if failures:
        print("FAILED:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("All selected figures regenerated.")


if __name__ == "__main__":
    main()
