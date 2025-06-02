import os
import pandas as pd
import zstandard as zstd
import json
from datetime import datetime

def decompress_zst_file(zst_path, output_jsonl_path):
    with open(zst_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with open(output_jsonl_path, 'wb') as out:
            out.write(dctx.decompress(compressed.read()))

def load_ohlcv_from_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 't' in entry and 'o' in entry:
                timestamp = datetime.fromtimestamp(entry['t'] / 1000.0)
                data.append({
                    'datetime': timestamp,
                    'open': entry['o'],
                    'high': entry['h'],
                    'low': entry['l'],
                    'close': entry['c'],
                    'volume': entry['v'],
                })
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df 