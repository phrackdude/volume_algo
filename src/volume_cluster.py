import pandas as pd

# Aggregates 1-min data into 15-minute blocks, then finds high-volume clusters
def identify_volume_clusters(df, volume_multiplier=3):
    # 15-minute resample
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    avg_volume = df_15m['volume'].mean()
    threshold = volume_multiplier * avg_volume
    clusters = df_15m[df_15m['volume'] >= threshold].copy()
    return clusters 