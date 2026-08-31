# Data Provenance

## BOSCH plasma-etch dataset

- Title: *A Multi-Model Dataset for BOSCH Plasma-Etching: Optical Emission Spectra, Process Parameters, and Wafer Measurements for Data-Driven Plasma Modeling*
- Authors: Mudassir Ali Sayyed, Tom Seifert, Stephan Zieger, Simeon Schwarzenberg, Aditya Deshmukh, Micha Haase, and Jan Langer
- Repository: Zenodo
- Record: https://zenodo.org/records/17122442
- DOI: https://doi.org/10.5281/zenodo.17122442
- Publication date: 2025-09-15
- Access date: 2026-08-30
- License: CC BY 4.0
- Local raw-data directory: `data/raw/zenodo_17122442/`
- Source metadata snapshot: `data/raw/zenodo_17122442/zenodo_record.json`

Raw source files are downloaded without modification. The following modeling-core files have been downloaded and verified against the MD5 checksums published by Zenodo:

| File | Source size (bytes) | Zenodo MD5 |
|---|---:|---|
| `Readme.pdf` | 359,703 | `f9a9bd323bd9e227a486249a631e8468` |
| `Wafer_layout.pdf` | 25,434 | `9bb916f316fadb02caad704f7cb2f09d` |
| `Lot_status.xlsx` | 11,954 | `339beca13f321dc3af244bf2d2ce284c` |
| `Si_Oxide_etch_89_points.csv` | 647,752 | `446e75b040eea37b634eeb8f763a62fc` |
| `Si_Oxide_etch_9_points.csv` | 46,954 | `78515caf25e29e558e1859b92f8a4827` |
| `Process_data.nc` | 8,819,467 | `4567d24ec2125102a2e5129203ba31fa` |
| `Dictionary_process.nc` | 88,084 | `0dde5a3a913eb1fa8512ef2f8748fb34` |
| `Dictionary_OES.nc` | 89,920 | `e7979670371604ff6f46d75dc3221cb0` |

## OES extension

The ten daily OES NetCDF files (`Day_2024_*.nc`) total approximately 7.90 GB and are outside
the scope of this benchmark. The download script supports them through `--include-oes`, but
the reported models use only process telemetry, context, and 89-site metrology.

## Integrity notes

- The raw directory is excluded from Git. Run `python scripts/download_zenodo.py` from the repository root to retrieve the compact modeling-core files directly from Zenodo and verify every published checksum. Pass `--include-oes` only when the approximately 7.9 GB OES extension is required.
- Downloaded-file checksums verify byte integrity only. They do not establish measurement quality, model validity, or causal interpretation.
- Process telemetry, chamber context, direct measurements, interpolated measurements, and derived targets must remain explicitly distinguished in downstream data dictionaries and models.

## Initial data-quality observations

- `Process_data.nc` contains 96 wafer groups. `Si_Oxide_etch_89_points.csv` contains 88 wafers with 89 rows each, so the current full-map supervised cohort is 88 wafers after an explicit key-based join.
- The ten wafers from 2024-07-02 contain 44 process channels; the other 86 wafers contain the 31-channel common subset. The benchmark uses the 31-channel intersection.
- In the 89-point data, pre-etch oxide thickness was directly measured at 15 locations per wafer and interpolated to the remaining locations with inverse-distance weighting. The `postox_thickness_nan` column retains 157 failed post-etch fits as `N/A`; `postox_thickness` contains their interpolated replacements.
- The published 89-point table satisfies `si_etch = stepheight - postox_thickness`, whereas the published 9-point table satisfies `si_etch = stepheight - oxide_etch`. The official Readme does not explain the inconsistent definitions. This benchmark uses only the 89-point definition.

For the canonical experiment hierarchy, field-level schemas, join rules, measurement lineage, target views, and leakage constraints, see [`DATA_STRUCTURE.md`](DATA_STRUCTURE.md).
