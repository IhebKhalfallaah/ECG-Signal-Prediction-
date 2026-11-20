# Data

The raw databases are **not** included in this repository (≈ 450 MB). Both are freely available on PhysioNet.

| Database | Role in this project | Records used | Link |
|---|---|---|---|
| PTB Diagnostic ECG Database v1.0.0 | Healthy controls → class `0 · Sain` | 80 | https://physionet.org/content/ptbdb/1.0.0/ |
| European ST-T Database v1.0.0 | Ischemic patients → classes `1 · angine` / `2 · infactus` | 31 | https://physionet.org/content/edb/1.0.0/ |

## Layout expected by the notebooks

```
data/
├── raw/                                     # not tracked by git
│   ├── ptb-diagnostic-ecg-database-1.0.0.zip
│   └── european-st-t-database-1.0.0.zip
├── work/                                    # created by the notebooks (extracted records, feature CSVs)
├── labels.csv                               # record → class
└── sample/e0103.{dat,hea,atr}               # demo record for the web app
```

Download the two zip archives ("Download the ZIP file" on each PhysioNet page) into `data/raw/`.

## `labels.csv`

One row per record used for training:

| Column | Description |
|---|---|
| `record` | WFDB record name (`e0103`, `s0273lre`, …) |
| `label` | `0` Sain (healthy) · `1` angine (angina) · `2` infactus (infarction) |
| `label_name` | Label as displayed by the app |
| `source` | Originating database |

Class counts: 80 × Sain, 20 × angine, 11 × infactus. Labels 1/2 are per-patient diagnoses of the European ST-T
records; healthy labels come from the PTB control group.

## Sample record

`sample/e0103` is record e0103 of the European ST-T Database (2 leads: V4 and MLIII, 250 Hz, 30 min).
Upload its `.dat`, `.hea` and `.atr` files in the web app to try a prediction.

## Citations

- Bousseljot R, Kreiseler D, Schnabel A. *Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet.* Biomedizinische Technik, 40(S1):317, 1995.
- Taddei A, Distante G, Emdin M, Pisani P, Moody GB, Zeelenberg C, Marchesi C. *The European ST-T Database: standard for evaluating systems for the analysis of ST-T changes in ambulatory electrocardiography.* European Heart Journal 13:1164-1172, 1992.
- Goldberger A, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23):e215–e220, 2000.
