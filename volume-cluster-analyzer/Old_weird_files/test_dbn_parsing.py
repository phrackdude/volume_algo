#!/usr/bin/env python3

import sys
import os
sys.path.append('src')
from src.parse_zst_ohlcv import parse_dbn_with_databento

def main():
    # Path to the DBN.ZST file
    dbn_path = "data/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst"
    
    # Test the DBN parsing function
    if os.path.exists(dbn_path):
        print(f"Testing DBN parsing with file: {dbn_path}")
        try:
            df = parse_dbn_with_databento(dbn_path)
            print("\n" + "="*60)
            print("FINAL PARSED DATA SAMPLE")
            print("="*60)
            print(df.head(10))
            print(f"\nShape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Save a sample for inspection
            sample_path = "data/parsed_sample.csv"
            df.head(100).to_csv(sample_path)
            print(f"\nSample data saved to: {sample_path}")
            
        except Exception as e:
            print(f"Error testing DBN parsing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"DBN file not found: {dbn_path}")

if __name__ == "__main__":
    main() 