from src.parse_zst_ohlcv import decompress_zst_file, load_ohlcv_from_jsonl
from src.volume_cluster import identify_volume_clusters
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Paths
zst_path = '../GLBX-20250602-C83SEA9BN7/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst'
jsonl_path = 'data/output.jsonl'

# Step 1: Decompress
print("Decompressing .zst...")
decompress_zst_file(zst_path, jsonl_path)

# Step 2: Load Data
print("Loading JSONL...")
df = load_ohlcv_from_jsonl(jsonl_path)
print(df.head())

# Step 3: Identify clusters
print("Identifying clusters...")
clusters = identify_volume_clusters(df)
print(clusters)

# Save results for inspection
clusters.to_csv("data/volume_clusters.csv")
print(f"Volume clusters saved to data/volume_clusters.csv")

# Number of clusters found
print(f"Found {len(clusters)} high volume clusters out of {len(df.resample('15T'))} 15-minute periods") 