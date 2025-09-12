# Volume Cluster Analyzer

A Python tool for identifying and analyzing high-volume clusters in OHLCV market data.

## Overview

This project analyzes 1-minute OHLCV (Open, High, Low, Close, Volume) data from financial markets to identify periods of unusually high trading volume. These volume clusters often represent significant market events or institutional activity that can provide valuable trading insights.

## Features

- Decompresses and parses zstd-compressed JSONL OHLCV data
- Aggregates 1-minute data into customizable time periods (default: 15 minutes)
- Identifies high-volume clusters based on configurable thresholds
- Analyzes price action before and after volume clusters
- Visualizes volume distribution and price action around clusters
- Saves analysis results for further investigation

## Directory Structure

```
volume-cluster-analyzer/
├── data/                   # Data directory for input and output files
├── notebooks/              # Jupyter notebooks for exploratory analysis
│   └── volume_cluster_explorer.py
├── src/                    # Source code
│   ├── __init__.py
│   ├── parse_zst_ohlcv.py  # OHLCV data parsing functions
│   └── volume_cluster.py   # Volume cluster detection logic
├── run_analysis.py         # Main script to run the analysis
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Usage

1. Place your zstandard-compressed OHLCV JSONL file in the `data` directory.
2. Run the analysis script:

```bash
python run_analysis.py
```

3. Check the `data` directory for output files:
   - `volume_clusters.csv`: The identified high-volume clusters
   - `volume_distribution.png`: Visualization of volume distribution
   - `price_volume_clusters.png`: Price chart with clusters highlighted
   - `cluster_analysis.csv`: Detailed analysis of price action around clusters

## Exploratory Analysis

For more in-depth exploration, run the Jupyter notebook or the Python script in the notebooks directory:

```bash
cd notebooks
python volume_cluster_explorer.py
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- zstandard
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```
