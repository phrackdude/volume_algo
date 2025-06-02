#!/usr/bin/env python3

import sys
import os
sys.path.append('src')
import databento as db
import pandas as pd

def analyze_symbols():
    # Path to the DBN.ZST file
    dbn_path = "data/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst"
    
    if os.path.exists(dbn_path):
        print(f"Analyzing symbols in: {dbn_path}")
        try:
            # Read the DBN file
            dbn_store = db.read_dbn(dbn_path)
            df = dbn_store.to_df()
            
            print(f"Total records: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Analyze symbols
            print("\n=== SYMBOL ANALYSIS ===")
            symbol_counts = df['symbol'].value_counts()
            print(f"Unique symbols: {len(symbol_counts)}")
            print("\nSymbol distribution:")
            print(symbol_counts.head(10))
            
            # Check price ranges by symbol
            print("\n=== PRICE ANALYSIS BY SYMBOL ===")
            for symbol in symbol_counts.head(5).index:
                symbol_data = df[df['symbol'] == symbol]
                print(f"\nSymbol: {symbol}")
                print(f"  Records: {len(symbol_data)}")
                print(f"  Price range: {symbol_data['low'].min():.2f} - {symbol_data['high'].max():.2f}")
                print(f"  Sample prices: Open={symbol_data['open'].iloc[0]:.2f}, High={symbol_data['high'].iloc[0]:.2f}, Low={symbol_data['low'].iloc[0]:.2f}, Close={symbol_data['close'].iloc[0]:.2f}")
            
            # Identify spread contracts (contain hyphen)
            spread_symbols = df[df['symbol'].str.contains('-', na=False)]['symbol'].unique()
            outright_symbols = df[~df['symbol'].str.contains('-', na=False)]['symbol'].unique()
            
            print(f"\n=== CONTRACT TYPE ANALYSIS ===")
            print(f"Spread contracts: {len(spread_symbols)}")
            print(f"Outright contracts: {len(outright_symbols)}")
            
            if len(spread_symbols) > 0:
                print(f"\nSpread symbols: {spread_symbols[:5]}")
                spread_data = df[df['symbol'].str.contains('-', na=False)]
                print(f"Spread price range: {spread_data['low'].min():.2f} - {spread_data['high'].max():.2f}")
                print(f"Spread records: {len(spread_data)}")
            
            if len(outright_symbols) > 0:
                print(f"\nOutright symbols: {outright_symbols[:5]}")
                outright_data = df[~df['symbol'].str.contains('-', na=False)]
                print(f"Outright price range: {outright_data['low'].min():.2f} - {outright_data['high'].max():.2f}")
                print(f"Outright records: {len(outright_data)}")
                
        except Exception as e:
            print(f"Error analyzing symbols: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"DBN file not found: {dbn_path}")

if __name__ == "__main__":
    analyze_symbols() 