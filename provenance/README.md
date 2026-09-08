# Provenance

`extract_helpfiles.py` produced `data/helpfiles/` from the archived PROTEUS output of the paper grid:

```bash
python provenance/extract_helpfiles.py --grid /path/to/archived/runs
```

`/path/to/archived/runs` holds one directory per run with the full `runtime_helpfile.csv` (tab-separated, more than 300 columns, one row per time step). The script copies, for each of the 41 runs the figures use, only the columns the figure scripts read, with every row and every value as text, so the reduced files reproduce the figures exactly. The column lists are in the script. The run configuration (`init_coupler.toml`) and termination status (`status.txt`) of each run are copied next to the reduced helpfile.

The full output (about 3 GB) is archived on the data server of the Kapteyn Astronomical Institute, University of Groningen, and is available from the corresponding author.

`fig06_contraction_grid/data/cache_shrinking.npz` is built from the S1 runs in `data/helpfiles/` by `fig06_contraction_grid/compute_shrinking.py`; running that script on the reduced helpfiles reproduces the cache array by array.
