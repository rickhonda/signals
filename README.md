# signals

This repository contains a small Python-based signal analysis engine.

It provides utilities for converting event-style data into time-series signals, applying simple baselines, and computing derived values such as residuals and scores.

---

## Overview

The core workflow supported by this project is:

1. Take an input table of timestamped events
2. Aggregate events into fixed time bins to produce one or more time series
3. Apply baseline estimators to each series
4. Compute residuals and simple scoring metrics

Configuration is primarily driven by YAML files that describe how channels are constructed.

---

## Repository layout

```
.
├── LICENSE
├── README.md
├── channels.yaml
├── main.py
├── signals.py
├── pyproject.toml
├── uv.lock
├── signal_analysis_example.png
└── examples/
    └── kerberos/
        ├── README.md
        ├── channels.yaml
        ├── dc_security.csv
        ├── dc_4769.parquet
        ├── analyzed.parquet
        ├── alerts.parquet
        ├── parse_windows_eventlog_xml.py
        ├── signal_analysis_demo.ipynb
        ├── windows-xml.log
        └── scratch/
            ├── dc_4769.csv
            └── signal_analysis_example.ipynb
```

---

## channels.yaml (repository root)

The `channels.yaml` file at the repository root contains generic examples of channel definitions.

It is intended to:

- Illustrate the expected structure of channel configuration
- Document available fields and common patterns
- Serve as a starting point for new configurations

It is not automatically applied to the example analyses.

Example-specific channel definitions live alongside their corresponding data under `examples/`.

---

## Examples

The `examples/` directory contains self-contained analyses that use the core engine.

Each example directory includes:

- Its own `channels.yaml`
- Input data files
- Optional notebooks and intermediate outputs
- A README describing how the example was produced

See `examples/kerberos/README.md` for details on the Kerberos example.

---

## Status

This project is under active development.

Interfaces, configuration formats, and file layout may change as the code evolves.

---

## License

MIT License. See `LICENSE` for details.
