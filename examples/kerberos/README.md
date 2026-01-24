# Kerberos Example

This directory contains a small Kerberos-focused example using Windows security event data to demonstrate signal construction, baselining, and scoring.

---

## Where the data came from

The CSV data in this directory is derived from **public Kerberos-related Windows security logs** published by Splunk as part of their attack / detection datasets.

The data has been lightly processed and reduced to make it suitable for demonstration purposes.  
It is not intended to represent a real production environment.

---

## What order to run things

From the repository root, run:

```bash
python signals.py build   --input examples/kerberos/dc_security.csv   --channels examples/kerberos/channels.yaml   --out examples/kerberos/signals.parquet
```

After this completes, you can explore the results using the notebook in this directory or by loading the generated Parquet file into your own analysis workflow.

You can safely rerun this command after changing parameters in `channels.yaml`.

---

## Which channels.yaml applies

This example uses **only**:

```
examples/kerberos/channels.yaml
```

The channels defined there include simple count-based signals and a derived Kerberos ratio channel, along with conservative baselining and scoring choices.

Any `channels.yaml` at the repository root does not apply to this example.
