# Signals

A small command-line tool for building hop-based time-series signals from event data and running baseline-driven anomaly detection.

This project is intentionally minimal and exploratory, focusing on signal construction, baselining, and scoring rather than end-to-end ingestion or alerting.

---

## Example workflow

The commands below reproduce the example data files used in the included notebook screenshot.

### Data source

The raw data originates from Splunk's public attack dataset:

https://github.com/splunk/attack_data/tree/master/datasets/attack_techniques/T1558.003/unusual_number_of_kerberos_service_tickets_requested

The file `windows-xml.log` was parsed into a normalized event stream named `dc_security.csv`, which is used as input here. The parsing step is outside the scope of this repository; the focus is on signal analysis once events are tabular.

---

### Build signals

Generate hop-based time series from the event stream:

```bash
python signals.py build --input dc_security.csv --channels channels.yaml --hop 1m --window 5m --out signals.parquet
```

This produces `signals.parquet`, containing one or more time-indexed channels representing x(t).

---

### Detect anomalies

Run baseline estimation and anomaly scoring:

```bash
python signals.py detect --input signals.parquet --alpha 0.05 --score-window 240 --threshold 6 --cusum --out analyzed.parquet
```

The output file (`analyzed.parquet`) includes the baseline, residual, score, and optional CUSUM state for each channel.

---

## Example output

Below is a screenshot from `signal_analysis_example.ipynb`, showing a single channel with its raw values, baseline, residual, score, and CUSUM output.

![Signal analysis example](signal_analysis_example.png)

---

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this code, including for commercial purposes, provided that the original copyright notice and license text are included.

See the `LICENSE` file for full details.
