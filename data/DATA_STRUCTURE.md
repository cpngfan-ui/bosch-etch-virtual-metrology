# BOSCH Plasma-Etch Data Dictionary

This document describes the source files, experimental hierarchy, join rules, measurement
lineage, modeling cohort, and the 81-feature schema used in the first wafer-level benchmark.
Counts were checked against Zenodo record 17122442 on 2026-08-30. Source attribution,
checksums, and license information are in [PROVENANCE.md](PROVENANCE.md).

## Modeling unit

One supervised sample is one 200 mm silicon wafer and its complete BOSCH etch run. The sample
contains:

- process telemetry indexed by timestamp and channel;
- 100 repeated process cycles within the run;
- wafer-level conditioning and sequence context; and
- one post-run metrology map measured at 89 fixed coordinates.

For wafer $i$, the raw input and spatial response can be written as

\[
X_i=\{S_i(t,c),q_i\}\longrightarrow Y_i(x_j,y_j),\qquad j=1,\ldots,89,
\]

where $S_i$ is the process trace, $q_i$ is known context, and $Y_i$ is the final wafer
map. Optical emission spectra $O_i(t,\lambda)$ are available as an optional source extension
but are not used in this benchmark.

| Level | Meaning | Count |
| --- | --- | ---: |
| Lot/date group | Sequential wafers processed after one conditioning sequence | 10 |
| Planned wafer | One physical 200 mm wafer | 100 |
| Process run | One wafer with usable telemetry | 96 |
| Labeled wafer | One process run with a complete 89-site map | 88 |
| BOSCH cycle | One alternating etch/passivation unit | 100 per run |
| Timestamp row | One multichannel telemetry record | 3,193–3,835 per run |
| Common process channel | One monitored machine or sensor signal | 31 |
| Metrology site | One fixed wafer-plane coordinate | 89 per labeled wafer |

Timestamps, cycles, channels, and metrology sites are components of a wafer sample, not
independent wafer experiments.

## Experiment and cohort

The experiments used one SPTS Omega i2L DSi Rapier system. A lot is a sequence of wafers run
on that tool after one conditioning sequence; it is not a separate chamber. Wafers within a
lot were processed sequentially, about one minute apart, without intermediate chamber
cleaning. They therefore share conditioning, date, tool, and an evolving chamber history.

The experimental factors recorded at lot level are:

- conditioning surface: chuck, blank Si wafer, or blank SiO2 wafer;
- conditioning count: 1, 3, or 9 repetitions; and
- wafer order within the subsequent lot.

The design is unbalanced. Chuck and Si conditioning occur at counts 1, 3, and 9; SiO2 occurs
only at count 3. Every conditioning combination is also tied to one date and lot.

| Lot | Date | Conditioning | Process runs | Complete maps |
| ---: | --- | --- | ---: | ---: |
| 1 | 2024-07-02 | 3 × chuck | 10 | 9 |
| 2 | 2024-07-05 | 1 × chuck | 10 | 10 |
| 3 | 2024-07-09 | 9 × chuck | 10 | 10 |
| 4 | 2024-07-11 | 3 × Si | 10 | 10 |
| 5 | 2024-07-19 | 1 × Si | 10 | 10 |
| 6 | 2024-08-01 | 9 × Si | 10 | 10 |
| 7 | 2024-08-05 | 3 × SiO2 | 6 | 6 |
| 8 | 2024-08-07 | 3 × SiO2 | 10 | 10 |
| 9 | 2024-08-21 | 3 × chuck | 10 | 9 |
| 10 | 2024-08-22 | 3 × SiO2 | 10 | 4 |
| **Total** |  |  | **96** | **88** |

Four wafers in lot 7 were reported damaged and have no usable process records. Eight additional
process runs have no matched complete 89-site map:

```text
2024-07-02_07
2024-08-21_09
2024-08-22_05
2024-08-22_06
2024-08-22_07
2024-08-22_08
2024-08-22_09
2024-08-22_10
```

The source documentation does not give a missingness mechanism for those eight maps.

## Process timing

The documented recipe consists of a nominal 1 s ignition followed by 100 cycles. Each cycle
has a nominal 4.5 s SF₆ etch phase and 1.5 s C₄F₈ passivation phase. The active cycle sequence
therefore lasts about ten minutes. Process telemetry is recorded at about 5 Hz, giving roughly
30 timestamp rows per nominal cycle.

Observed traces contain 3,193–3,835 rows, with a median of 3,247. Ignition, acquisition
windows, phase transitions, and timestamp gaps account for the difference from a simple
100-cycle estimate. The dataset contains final metrology only; it does not contain silicon
depth after each intermediate cycle.

## Source files

| File | Role | Included in v1 |
| --- | --- | --- |
| `Readme.pdf` | Experiment, recipe, and metrology description | Yes |
| `Wafer_layout.pdf` | 200 mm wafer and 89-site/9-site layouts | Yes |
| `Lot_status.xlsx` | Dates, wafer counts, lots, and conditioning codes | Yes |
| `Process_data.nc` | Process telemetry for 96 runs | Yes |
| `Dictionary_process.nc` | Decoder for encoded process values | Yes |
| `Si_Oxide_etch_89_points.csv` | 89-site metrology for 88 wafers | Yes |
| `Si_Oxide_etch_9_points.csv` | Separate sparse metrology campaign | Audited, not modeled |
| `Dictionary_OES.nc` | Decoder for encoded OES values | Audited, not modeled |
| `Day_2024_*.nc` | Ten daily OES files, about 7.9 GB total | No |

Raw files are excluded from Git. `python scripts/download_zenodo.py` downloads the compact
files directly from Zenodo and checks their published MD5 hashes.

## Process telemetry schema

`Process_data.nc` contains 96 groups named as follows:

```text
Day_YYYY_MM_DD_Wafer_NN
```

Each group contains:

| Object | Shape | Meaning |
| --- | --- | --- |
| `feature` | \((C_i,)\) | Ordered raw channel names |
| `times` | \((T_i,)\) | Numeric within-run timestamps |
| `data` | \((T_i,C_i)\) | Encoded 16-bit process array |
| `time` | \((T_i,)\) | NetCDF dimension scale |

The `data` values are indices into the 49,290-element `float32` array in
`Dictionary_process.nc`:

```python
decoder = dictionary_process["data"][:]
encoded = wafer_group["data"][:]
decoded = decoder[encoded]
```

All used indices are within bounds, and the decoded values are finite.

| Audited property | Value |
| --- | ---: |
| Process groups | 96 |
| Total timestamp rows | 313,169 |
| Median rows per run | 3,247 |
| Median adjacent interval | about 0.2 s |
| Common channels | 31 |
| Channels on 2024-07-02 | 44 |
| Runs using the common 31-channel schema | 86 |

In 69 runs, one isolated final timestamp appears about 41–45 seconds after the main regular
block. Feature extraction uses explicit timestamps and retains the longest contiguous block.
Calendar date and wafer identity come from the group name; `times` is treated as a within-run
coordinate.

### Common channel dictionary

The release does not provide complete engineering units, controller definitions, or a mapping
from numbered gas channels to chemical species. I treat all `MV` fields as monitored values or
readbacks rather than commanded setpoints.

| Physical group | Common channels |
| --- | --- |
| Endpoint and vacuum | `EpdIntensity`, `ForeLinePressure`, `Pressure` |
| Gas flow | `Gas1Flow`, `Gas2Flow`, `Gas3Flow`, `Gas4Flow`, `Gas5Flow`, `Gas7Flow`, `Gas8Flow` |
| Thermal | `Heater1Temp`, `Heater2Temp`, `Heater3Temp`, `Heater4Temp` |
| Backside helium | `HeliumBPFlow`, `HeliumBPPressure` |
| Platen RF | `PlatenDcBias`, `PlatenRFLoadCapacitor`, `PlatenRFLoadPower`, `PlatenRFPeakToPeak`, `PlatenRFReflectedPower`, `PlatenRFTuningCapacitor` |
| Source RF | `SourceRFLoadPower`, `SourceRFPeakToPeak`, `SourceRFReflectedPower`, `SourceRFTuningCapacitor` |
| Source RF2 | `SourceRF2LoadPower`, `SourceRF2PeakToPeak`, `SourceRF2ReflectedPower`, `SourceRF2TuningCapacitor` |
| Other | `moriInnerCurrent` |

Every name above has the raw prefix `Stat3_Etch_MV_`. Four common channels are constant zero in
the audited snapshot: `Gas3Flow`, `Gas8Flow`, `SourceRF2LoadPower`, and
`SourceRF2ReflectedPower`.

The first ten runs include 13 additional channels that are absent on later dates:

```text
Gas6Flow
Heater5Temp, Heater6Temp, Heater7Temp, Heater8Temp
SourceRF2LoadCapacitor, SourceRFLoadCapacitor
ThermoCouple1Temp, ThermoCouple2Temp, ThermoCouple3Temp, ThermoCouple4Temp
attenuatorRatio, moriOuterCurrent
```

V1 uses the 31-channel intersection. Using first-date-only channels would introduce a
missingness pattern that directly identifies lot 1.

The raw process file does not contain cycle indices, phase labels, recipe-transition labels,
a paired setpoint table, or intermediate etch depth. Cycle segmentation is therefore a
derived, target-independent alignment.

## Optical emission spectra

OES records light emitted by the plasma. For a run, the source representation is a
time-by-wavelength matrix with 3,648 wavelength bins spanning approximately 185.89–883.97 nm
at a nominal 25 Hz sampling rate. The 3,648 columns are wavelength positions, not wafer sites
or process timestamps.

The ten daily OES files total about 7.9 GB and were not used in v1. A future OES comparison
should join by wafer key, fit wavelength reduction inside training folds, and compare
process-only with process-plus-OES on the same matched cohort.

## Metrology schema

`Si_Oxide_etch_89_points.csv` contains 7,832 rows: 88 wafers with 89 spatial sites each. Every
wafer uses the same set of unique \((X,Y)\) coordinates.

| Spatial property | Value |
| --- | ---: |
| Wafer diameter | 200 mm |
| X range | -95,000 to 95,000 µm |
| Y range | -95,000 to 95,000 µm |
| Main grid spacing | 19,000 µm |
| Sites per wafer | 89 |

| Column | Unit | Grain | Meaning |
| --- | --- | --- | --- |
| `experiment_key` | none | wafer | `YYYY-MM-DD_NN` join key |
| `lot_number` | none | wafer | Lot 1–10 |
| `wafer_number` | none | wafer | Sequential order within the lot |
| `X`, `Y` | µm | site | Wafer-plane coordinates |
| `preox_thickness` | µm | site | Pre-etch SiO2 mask thickness, including interpolation |
| `postox_thickness` | µm | site | Remaining SiO2, including completion of failed fits |
| `postox_thickness_nan` | µm | site | Original post-etch result with failed fits left missing |
| `stepheight` | µm | site | Height from remaining mask top to etched Si surface |
| `oxide_etch` | µm | site | Derived SiO2 consumption |
| `si_etch` | µm | site | Derived cumulative silicon etch depth |

The published 89-site fields satisfy:

\[
\texttt{oxide\_etch}=\texttt{preox\_thickness}-\texttt{postox\_thickness},
\]

\[
\texttt{si\_etch}=\texttt{stepheight}-\texttt{postox\_thickness}.
\]

`si_etch` is the silicon removed during the run, not the remaining thickness of the silicon
wafer.

### Measurement lineage

| Field | Lineage | Limitation |
| --- | --- | --- |
| `preox_thickness` | 15 direct pre-etch sites per wafer, then inverse-distance interpolation | The CSV does not retain a row-level direct/interpolated flag |
| `postox_thickness_nan` | Original post-etch fit | Contains 157 failed fits |
| `postox_thickness` | Original fits plus inverse-distance completion | Completed sites occur across 20 wafers |
| `stepheight` | Profilometer measurement | All 7,832 published rows are present |
| `oxide_etch` | Derived from pre- and post-oxide thickness | Inherits both inputs' uncertainty |
| `si_etch` | Derived from step height and completed post-oxide thickness | Inherits metrology and completion uncertainty |

The labels come from physical wafers and physical metrology, but they are measurement
references rather than error-free values. The reproducible post-oxide completion flag is:

```python
postox_was_interpolated = postox_thickness_nan.isna()
```

The first published spatial row is:

```text
experiment_key       2024-07-02_01
lot_number           1
wafer_number         1
X                    -19000 µm
Y                    -95000 µm
preox_thickness      1.002226171 µm
postox_thickness     0.5259 µm
stepheight           52.989 µm
oxide_etch           0.476326171 µm
si_etch              52.4631 µm
```

This row is one coordinate on one wafer. The 89 rows sharing its `experiment_key` form the
spatial response for that wafer.

## Process–metrology join

The process and metrology files encode wafer identity differently:

```text
NetCDF group   Day_2024_07_05_Wafer_01
CSV key        2024-07-05_01
Joined key     2024-07-05_01
```

The join procedure:

1. parses date and wafer number from the NetCDF group;
2. constructs `experiment_key = YYYY-MM-DD_NN`;
3. checks uniqueness of process keys;
4. checks that each map has 89 unique coordinates;
5. groups site rows into a wafer map;
6. joins process and metrology by `experiment_key`; and
7. records unmatched keys and exclusion reasons.

All 88 metrology keys match process groups. The unmatched records are the eight process-only
runs listed above.

## V1 target and model table

The first benchmark uses:

| Item | Definition |
| --- | --- |
| Cohort | 88 matched wafer runs |
| Input | Complete-run telemetry summaries plus known context |
| Target | Arithmetic mean of 89 published `si_etch` site values |
| Group | Lot/date, 10 groups |
| Prediction time | After the run and before metrology |
| OES | Excluded |
| 9-site campaign | Excluded |
| First-date-only channels | Excluded |

For wafer (i),

\[
y_i=\frac{1}{89}\sum_{j=1}^{89}\texttt{si\_etch}_{ij}.
\]

This is a simple site average, not an area-weighted wafer integral. The resulting model matrix
has shape \(88\times81\), and the target vector has length 88.

## The 81-feature schema

Feature construction is deterministic and does not use metrology values for segmentation,
selection, or calculation.

| Block | Count | Construction |
| --- | ---: | --- |
| Cycle response | 56 | 14 channels × 4 cycle summaries |
| Timing | 10 | Phase, period, anomaly, active-span, and startup summaries |
| Slow state | 10 | 5 channels × 2 run summaries |
| Context | 5 | Conditioning and wafer sequence |
| **Total** | **81** | One vector per wafer run |

### Timestamp cleanup and cycle segmentation

For each run, the code requires increasing timestamps, estimates the median sampling interval,
splits at gaps larger than `max(1.0 s, 5 × median interval)`, and keeps the longest block.

Cycle structure is detected from three monitored signals:

| Role | Signal |
| --- | --- |
| Short-phase anchor | `Stat3_Etch_MV_Gas4Flow` |
| Long-phase proxy | `Stat3_Etch_MV_Gas5Flow` |
| Source-active proxy | `Stat3_Etch_MV_SourceRFLoadPower` |

Gas4 and Gas5 thresholds are the midpoint between the within-run 5th and 95th percentiles.
High components shorter than 0.6 s are rejected. A trace passes when it contains exactly 100
Gas4 anchors, at least one Gas5 component in every anchored cycle, and a source-active
component. All 96 traces pass these checks.

Main cycle statistics use cycles 2–99, leaving 98 stable cycles. Gas4 and Gas5 remain anonymous
phase proxies because the source does not publish their chemical identities.

### Cycle-response features

For stable cycle (c) and channel (k), let (s_{ck}) and (l_{ck}) be the time-weighted
means during the detected short and long components. Define

\[
o_{ck}=(s_{ck}+l_{ck})/2,\qquad d_{ck}=l_{ck}-s_{ck}.
\]

Each selected channel contributes:

| Suffix | Definition |
| --- | --- |
| `level_median` | Median of \(o_{ck}\) across stable cycles |
| `phase_contrast_median` | Median of \(d_{ck}\) |
| `phase_contrast_mad` | Median absolute deviation of \(d_{ck}\) |
| `late_minus_early` | Mean of the last ten \(o_{ck}\) values minus the first ten |

The 14 selected channels are:

| Block | Channels |
| --- | --- |
| Gas delivery | `Gas4Flow`, `Gas5Flow` |
| Pressure and vacuum | `ForeLinePressure`, `Pressure` |
| Backside thermal coupling | `HeliumBPFlow` |
| Platen RF | `PlatenDcBias`, `PlatenRFLoadPower`, `PlatenRFPeakToPeak`, `PlatenRFReflectedPower`, `PlatenRFLoadCapacitor`, `PlatenRFTuningCapacitor` |
| Source RF | `SourceRFLoadPower`, `SourceRFPeakToPeak`, `SourceRFReflectedPower` |

The raw prefix for every channel is `Stat3_Etch_MV_`.

### Timing features

| Feature | Definition |
| --- | --- |
| `short_duration_median_s` | Median stable short-component duration |
| `short_duration_iqr_s` | IQR of stable short durations |
| `long_duration_median_s` | Median summed long-component duration |
| `long_duration_iqr_s` | IQR of summed long durations |
| `cycle_period_median_s` | Median interval between consecutive stable anchors |
| `cycle_period_iqr_s` | IQR of stable anchor intervals |
| `extra_long_components` | Gas5 components across 100 cycles minus 100 |
| `abnormal_cycle_intervals` | Count with \(|period-6.0|>0.3\) s |
| `source_active_span_s` | Duration of the longest source-active component |
| `startup_interval_s` | Start of anchor 2 minus start of anchor 1 |

`startup_interval_s` is a transition proxy, not a direct measurement of the nominal ignition.

### Slow-state and context features

The slow-state channels are `Heater2Temp`, `Heater3Temp`, `Heater4Temp`, `Gas2Flow`, and
`Gas7Flow`. Each contributes its median active level and last-ten minus first-ten cycle drift.
Gas2 and Gas7 remain anonymous flow readbacks.

The five context columns are:

- `context__conditioning_count`;
- `context__conditioning_surface_chuck`;
- `context__conditioning_surface_si`;
- `context__conditioning_surface_sio2`; and
- `context__wafer_order`.

Lot number, date, and `experiment_key` are grouping or identity fields, not predictors.

The schema uses 19 unique telemetry channels. Twelve common channels are excluded: endpoint
intensity, Gas1, the four constant-zero signals, Heater1, helium pressure, the remaining RF2
signals, source tuning capacitor, and inner current. Their exclusion is a compact engineering
choice for 88 labeled wafers, not a conclusion that they are physically irrelevant.

## Generated tables

| File | Rows | Contents |
| --- | ---: | --- |
| `all_predictors.csv` | 96 | The 81 predictors for every process run |
| `cohort.csv` | 96 | Lot, conditioning, sequence, and target availability |
| `feature_qa.csv` | 96 | Timestamp and segmentation checks |
| `feature_table.csv` | 88 | IDs, group, scalar target, and 81 predictors |
| `target_audit.csv` | 88 | Site count, spatial summaries, and target-lineage sensitivity |
| `feature_metadata.json` | 1 | Feature views and task columns |
| `prepare_manifest.json` | 1 | Counts, configuration, hashes, and seed |

`feature_table.csv` has 85 columns: `experiment_key`, `lot_number`, `wafer_order`,
`mean_si_etch_um`, and 81 predictors. Except for timing fields, decoded telemetry units are not
available in the release and are not inferred during preparation.

## Validation and leakage controls

The outer evaluation holds out one complete lot at a time. Hyperparameters and learned
preprocessing are selected through an inner leave-one-training-lot-out loop. Because date and
lot are one-to-one, this is also a date-held-out transfer test.

The following operations stay inside their training fold:

- imputation, variance filtering, and scaling;
- feature selection and process-signal reduction;
- OES wavelength selection or reduction;
- any target-map spatial basis;
- hyperparameter selection; and
- interval calibration.

Leakage checks prevent site-level random splitting, duplication of a wafer label across cycles
or timestamps, use of identity fields as predictors, target-derived segmentation, and fitting
preprocessing before the split. The 89 rows from one wafer always remain together.

## Other target formulations

The current target is a wafer-level mean. A spatial extension can instead predict the complete
89-element map. The dataset would then have 88 map labels with 89 components each, not 7,832
independent runs. A reduced-rank basis such as target PCA must be fitted inside each outer
training fold before score prediction and map reconstruction.

Other possible responses include mean oxide etch, selectivity, within-wafer non-uniformity,
center-to-edge difference, radial slope, and spatial basis coefficients. Each derived response
needs a fixed formula, spatial mask, unit, and weighting rule before modeling.

The separate 9-site file is not combined with the 89-site cohort. It has 684 rows: 75 identified
wafers with nine rows each plus nine rows without wafer keys. It also uses a different published
relationship, `si_etch = stepheight - oxide_etch`, while the 89-site table uses
`si_etch = stepheight - postox_thickness`. The source documentation does not reconcile the two
definitions.

## Operational boundary

| Category | Example | Role in this project | Directly controllable |
| --- | --- | --- | --- |
| Recipe setpoint | Gas-flow, RF-power, pressure, or timing command | Not available as a paired v1 input table | Yes, within hardware limits |
| Chamber state | Wall condition, plasma density, thermal state | Latent physical condition | Usually no |
| Sensor/readback | Flow MV, pressure MV, reflected power, temperature | Model input after the run | No |
| OES | Time-by-wavelength emission | Optional observation of plasma state | No |
| Context | Conditioning and wafer order | Known model input | Mixed |
| Metrology | Step height, oxide thickness, silicon etch | Response | No |

V1 supports prediction after a run has completed and before metrology is available. Pre-run
recipe optimization would require explicitly recorded and sufficiently varied setpoints, a
designed experiment, pre-run state information, and confirmation on the processing tool.
