# Figure data and scripts: Super-Earth Interiors Shrink by About 10% as They Crystallise

Data and scripts that reproduce every figure of the paper *Super-Earth Interiors Shrink by About 10% as They Crystallise* (Lichtenberg et al. 2026). Archived on Zenodo: [10.5281/zenodo.22663462](https://doi.org/10.5281/zenodo.22663462) (all versions; the first release, v26.09.08, is [10.5281/zenodo.22663463](https://doi.org/10.5281/zenodo.22663463)). The repository is self-contained: one command regenerates all 19 figures from the data it holds. Nothing is downloaded, no environment variable is read, and no file outside the repository is touched.

## Quick start

```bash
conda env create -f environment.yml
conda activate superearth-figures
python make_figures.py
```

The PDFs are written to `figures/` under the file names used in the paper. `python make_figures.py --only redox volatiles` runs a subset. Figure 1 is a TikZ drawing; `make_figures.py` compiles it when `latexmk` is on the PATH and otherwise copies the committed `figures/proteus_loop.pdf` into the output directory.

The plotting needs Python 3.12 with numpy, matplotlib and pandas; `requirements.txt` pins the versions that produced the figures in the paper, and `environment.yml` installs them from PyPI into a conda environment (`pip install -r requirements.txt` in any Python 3.12 works as well). The PyPI wheel of matplotlib bundles FreeType 2.6.1, and the figures in the paper were rendered with it; a matplotlib built against another FreeType version, for example the conda-forge package, places glyphs a fraction of a point differently.

## Layout

Each figure directory holds one plotting script and the data it reads. A directory name lists the numbers of the figures it produces, in order of appearance in the paper.

| Figures | Directory | Paper files | Script | Data read |
|---|---|---|---|---|
| 1 | `fig01_proteus_loop/` | `proteus_loop.pdf` | `proteus_loop.tex` (TikZ) | none, schematic |
| 2, 13 | `fig02_13_zalmoxis_literature_mr/` | `zalmoxis_literature_mr.pdf`, `zalmoxis_literature_profiles.pdf` | `plot_literature_mr.py` | `zalmoxis_grids/`, `magrathea/`, `reference_data/` |
| 3 | `fig03_calliope_atmodeller/` | `calliope_atmodeller.pdf` | `plot_calliope_atmodeller.py` | `data/fig3_grid.csv` |
| 4 | `fig04_chili_validation/` | `chili_validation.pdf` | `plot_chili_validation.py` | `data/cache_chili.npz` |
| 5 | `fig05_phi_sweep_m1/` | `phi_sweep_m1.pdf` | `plot_phi_sweep.py` | `data/zalmoxis_phi_sweep_m1_mantle.csv` |
| 6 | `fig06_contraction_grid/` | `contraction_grid.pdf` | `plot_contraction_grid.py` | `data/cache_shrinking.npz`, built by `compute_shrinking.py` from the S1 runs in `data/helpfiles/` |
| 7 | `fig07_redox/` | `redox.pdf` | `plot_redox.py` | the S2 runs in `data/helpfiles/`, `data/phi040_window.csv` |
| 8 | `fig08_volatiles/` | `volatiles.pdf` | `plot_volatiles.py` | the S1 dynamic and S3 runs in `data/helpfiles/` |
| 9 | `fig09_robustness/` | `robustness.pdf` | `plot_robustness.py` | `data/robustness_summary.csv` |
| 10, 11, 12 | `fig10_11_12_zalmoxis_first_principles/` | `zalmoxis_spheres.pdf`, `zalmoxis_lane_emden.pdf`, `zalmoxis_conservation_convergence.pdf` | `plot_first_principles_validation.py` | `data/*.npz` |
| 14, 15, 16 | `fig14_15_16_aragog_first_principles/` | `aragog_conduction.pdf`, `aragog_conservation.pdf`, `aragog_transient.pdf` | `plot_aragog_first_principles.py` | `data/*.npz` |
| 17, 18 | `fig17_18_aragog_spider_parity/` | `aragog_spider_const.pdf`, `aragog_spider_static.pdf` | `plot_aragog_spider_parity.py` | `data/cache_const_anchor.npz`, `data/cache_static_paleos.npz` |
| 19 | `fig19_calliope_atmodeller_outgassing/` | `calliope_atmodeller_outgassing.pdf` | `plot_outgassing.py` | `data/cache_outgassing.npz` |

Shared directories:

- `data/helpfiles/<run>/`: the PROTEUS output of the 41 coupled simulation runs that Figures 6, 7 and 8 read. Each run directory holds `runtime_helpfile.csv` (tab-separated, one row per time step, reduced to the columns the figure scripts read, all rows kept, values copied verbatim), the run configuration `init_coupler.toml`, and the status of the run when the output was archived, `status.txt`. Four S2 runs (`S2_m5_IWp2_fixed`, `S2_m5_IWp2_oxauth`, `S2_m5_IWp3_oxauth`, `S2_m5_IWp4_fixed`) were still advancing when archived; Figure 7 reads from them only the state interpolated at global melt fraction 0.40 and the values at the first time step, both inside the range every one of them covers.
- `provenance/extract_helpfiles.py`: the script that produced `data/helpfiles/` from the full PROTEUS output, with the per-run column lists. The full output (more than 300 columns per time step, about 3 GB) is archived on the Kapteyn Astronomical Institute data server.
- `style/`: the PROTEUS visual language for matplotlib (`proteus_mpl.py`, tokens, style sheets; Apache License 2.0, `style/proteus_assets/LICENSE-CODE`, from [FormingWorlds/proteus-visual-language](https://github.com/FormingWorlds/proteus-visual-language)) with the bundled fonts Instrument Sans, Sora and Spline Sans Mono (SIL Open Font License, `style/proteus_assets/fonts/OFL.txt`).
- `figures/`: the output of `make_figures.py`, committed so the repository shows the result.

## Simulation runs in `data/helpfiles/`

| Set | Runs | Used by |
|---|---|---|
| S1 | `S1_m{1,3,5,10}_{dyn,stat}_IW4`: planet mass 1, 3, 5, 10 Earth masses, dynamic or static interior structure, IW+4 | Figure 6 (all 8), Figure 8 (the 4 dynamic runs) |
| S2 | `S2_m5_IW{m6..p5}_{fixed,oxauth}`: 5 Earth masses, imposed oxygen fugacity IW-6 to IW+5, fixed-fugacity and oxygen-conserving treatment | Figure 7 |
| S3 | `S3_T{1,3,4}_m{3,5,10}`: volatile-inventory treatments T1, T3, T4 at 3, 5, 10 Earth masses | Figure 8 |

Columns kept per run: S1, `Time R_int R_obs R_core Phi_global T_magma T_surf T_cmb R_solvus RF_depth P_surf M_mantle_solid M_mantle_liquid gravity fO2_shift_IW_derived` (Figure 8 reads `R_int`, `Phi_global` and `P_surf` from the dynamic runs); S2 fixed-fugacity, `Phi_global P_surf` plus the volume mixing ratio of H2O, H2, CO2, CO, O2, CH4, N2, S2, SO2, H2S and the total mass of H2O, CO2, CO, O2; S2 oxygen-conserving, `Phi_global fO2_shift_IW_derived`; S3, `R_int Phi_global P_surf`.

`fig07_redox/data/phi040_window.csv` holds the S2 fixed-fugacity runs at IW0 to IW+5 in a window around global melt fraction 0.40, from a separate set of runs of the same configurations with sulfur speciation resolved (in the S2 helpfiles the SO2 mixing ratio is floored at 1e-30). `plot_redox.py` reads the oxidising half of the speciation panel from this file and everything else from `data/helpfiles/`.

## Provenance of the cached validation data

| File | Content | Produced with |
|---|---|---|
| `fig02_13_zalmoxis_literature_mr/zalmoxis_grids/` | Zalmoxis mass-radius grids (PALEOS and [Seager et al. 2007](#references) equations of state) and interior profiles | Zalmoxis |
| `fig02_13_zalmoxis_literature_mr/magrathea/` | MAGRATHEA mass-radius relations and interior profiles ([Huang et al. 2022](#references)) | MAGRATHEA |
| `fig02_13_zalmoxis_literature_mr/reference_data/` | [Zeng et al. (2016)](#references) tabulated mass-radius relations | published tables |
| `fig03_calliope_atmodeller/data/fig3_grid.csv` | oxygen-fugacity offsets of CALLIOPE and atmodeller across magma temperature | CALLIOPE cross-module comparison (`scripts/cross_backend` in the CALLIOPE repository) |
| `fig04_chili_validation/data/cache_chili.npz` | melt-fraction and temperature curves of the CHILI community benchmark (github.com/projectcuisines/chili) and of the PROTEUS Earth and Venus reference runs | PROTEUS (`tools/plot_chili_comparison.py`) |
| `fig05_phi_sweep_m1/data/zalmoxis_phi_sweep_m1_mantle.csv` | interior radius of a 1 Earth-mass planet against mantle melt fraction | Zalmoxis |
| `fig09_robustness/data/robustness_summary.csv` | total contraction of the S5 and S6 runs | extracted from the PROTEUS S5 and S6 runs |
| `fig10_11_12_zalmoxis_first_principles/data/*.npz` | constant-density sphere, Lane-Emden polytrope and convergence tests | Zalmoxis (`plot_first_principles_validation.py --recompute` with the package installed) |
| `fig14_15_16_aragog_first_principles/data/*.npz` | conduction, energy-conservation and transient tests | Aragog (`plot_aragog_first_principles.py --recompute` with the package installed) |
| `fig17_18_aragog_spider_parity/data/*.npz` | Aragog and SPIDER on the same mesh and entropy profile | Aragog and a built SPIDER binary |
| `fig19_calliope_atmodeller_outgassing/data/cache_outgassing.npz` | equilibrium speciation from CALLIOPE and atmodeller | CALLIOPE and atmodeller |

## Software

The simulations and the cached validation data were produced with the software versions cited in the paper:

| Code | Version | Record |
|---|---|---|
| PROTEUS | v26.07.14 | [10.5281/zenodo.21358381](https://doi.org/10.5281/zenodo.21358381) |
| Aragog | v26.07.04 | [10.5281/zenodo.21196696](https://doi.org/10.5281/zenodo.21196696) |
| Zalmoxis | v26.07.13 | [10.5281/zenodo.21342353](https://doi.org/10.5281/zenodo.21342353) |
| PALEOS | v1.0.0 | [10.5281/zenodo.19000316](https://doi.org/10.5281/zenodo.19000316) (equation-of-state tables), [10.5281/zenodo.19221215](https://doi.org/10.5281/zenodo.19221215) (mass-radius tables), [Attia et al. (2026)](#references) |
| CALLIOPE | v26.07.03 | [10.5281/zenodo.21162734](https://doi.org/10.5281/zenodo.21162734) |
| atmodeller | v1.0.1 | [Bower et al. (2025)](#references) |
| AGNI | v1.9.4 | [10.5281/zenodo.15386789](https://doi.org/10.5281/zenodo.15386789), [Nicholls et al. (2025)](#references) |
| MORS | v26.07.12 | [10.5281/zenodo.21315171](https://doi.org/10.5281/zenodo.21315171) |
| ZEPHYRUS | v26.07.10 | [10.5281/zenodo.21301993](https://doi.org/10.5281/zenodo.21301993) |

## References

- Attia, M., Lichtenberg, T., Jungová, E., and Sastre, M. (2026). PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors. Astronomy and Astrophysics, in press. [doi:10.1051/0004-6361/202660790](https://doi.org/10.1051/0004-6361/202660790), [arXiv:2605.03741](https://arxiv.org/abs/2605.03741)
- Bower, D. J., Thompson, M. A., Hakim, K., Tian, M., and Sossi, P. A. (2025). Diversity of low-mass planet atmospheres in the C-H-O-N-S-Cl system with interior dissolution, nonideality, and condensation: application to TRAPPIST-1e and sub-Neptunes. The Astrophysical Journal, 995, 59. [doi:10.3847/1538-4357/ae1479](https://doi.org/10.3847/1538-4357/ae1479), [ADS](https://ui.adsabs.harvard.edu/abs/2025ApJ...995...59B)
- Huang, C., Rice, D. R., and Steffen, J. H. (2022). MAGRATHEA: an open-source spherical symmetric planet interior structure code. Monthly Notices of the Royal Astronomical Society, 513, 5256. [doi:10.1093/mnras/stac1133](https://doi.org/10.1093/mnras/stac1133), [ADS](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.5256H)
- Nicholls, H., Pierrehumbert, R., and Lichtenberg, T. (2025). AGNI: a radiative-convective model for lava planet atmospheres. The Journal of Open Source Software, 10, 7726. [doi:10.21105/joss.07726](https://doi.org/10.21105/joss.07726), [ADS](https://ui.adsabs.harvard.edu/abs/2025JOSS...10.7726N)
- Seager, S., Kuchner, M., Hier-Majumder, C. A., and Militzer, B. (2007). Mass-radius relationships for solid exoplanets. The Astrophysical Journal, 669, 1279. [doi:10.1086/521346](https://doi.org/10.1086/521346), [ADS](https://ui.adsabs.harvard.edu/abs/2007ApJ...669.1279S)
- Zeng, L., Sasselov, D. D., and Jacobsen, S. B. (2016). Mass-radius relation for rocky planets based on PREM. The Astrophysical Journal, 819, 127. [doi:10.3847/0004-637X/819/2/127](https://doi.org/10.3847/0004-637X/819/2/127), [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z)

## Citation and licence

Cite the paper and this record, [10.5281/zenodo.22663462](https://doi.org/10.5281/zenodo.22663462) (see `CITATION.cff`). The paper:

> Lichtenberg, T., Attia, M., Nicholls, H., Sastre, M., Bower, D. J., Stuitje, K., Pascal, F. C., Soucasse, L., Apai, D., Bos, P., Calder, R., Cesario, L., Dang, L., Decocq, E., van Dijk, M., Farhat, M., Hakim, K., Kimura, T., Kisvárdai, I., Krijt, S., Miguel, Y., Panagiotou, I., Postolec, E., Schlecker, M., Seager, S., Shahar, A., Shorttle, O., Sossi, P. A., and van Westrenen, W. (2026). Super-Earth Interiors Shrink by About 10% as They Crystallise.

The data, the figures and the scripts in this repository are released under the Creative Commons Attribution 4.0 International licence (`LICENSE`). The vendored PROTEUS visual-language code in `style/` keeps its Apache License 2.0 (`style/proteus_assets/LICENSE-CODE`) and the bundled fonts their SIL Open Font License (`style/proteus_assets/fonts/OFL.txt`).
