import os
import pandas as pd
import zstandard as zstd
import json
from datetime import datetime, timedelta
import struct
import numpy as np
import io
from .synthetic_data import generate_synthetic_ohlcv, save_synthetic_data

def decompress_zst_file(zst_path, output_jsonl_path):
    """Decompresses a ZST file to the specified output path."""
    print(f"Decompressing {zst_path}...")
    with open(zst_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with open(output_jsonl_path, 'wb') as out:
            out.write(dctx.decompress(compressed.read()))
    print(f"Decompressed to {output_jsonl_path} ({os.path.getsize(output_jsonl_path)} bytes)")

def parse_dbn_file(file_path):
    """Attempts to parse a DBN format file containing OHLCV data."""
    print(f"Attempting to parse DBN format file: {file_path}")
    
    with open(file_path, 'rb') as f:
        header = f.read(16)
        
        if not header.startswith(b'DBN'):
            print("Not a DBN file (missing DBN signature)")
            return None
            
        print(f"DBN header: {header}")
        
        # Try to extract metadata
        try:
            # Read the rest of the file
            content = f.read()
            
            # Search for OHLCV data patterns
            # This is a simplistic approach that looks for patterns in the binary data
            # that might represent timestamps and price data
            
            # Create a list to store parsed records
            records = []
            
            # Use a sliding window to find potential OHLCV records
            window_size = 40  # Approximate size of a record
            step_size = 20    # Step size for window
            
            pos = 0
            while pos < len(content) - window_size:
                chunk = content[pos:pos+window_size]
                
                # Look for potential timestamp patterns (increasing integers)
                if len(chunk) >= 8:
                    # Try to extract timestamp-like values (assuming 4 or 8 byte integers)
                    try:
                        timestamp_ms = int.from_bytes(chunk[0:8], byteorder='little')
                        
                        # If it looks like a reasonable timestamp (between 2020 and 2025)
                        if 1577836800000 < timestamp_ms < 1735689600000:  # 2020-01-01 to 2025-01-01
                            # Try to extract OHLCV values (assuming they're floating point)
                            try:
                                # Find reasonable values for OHLC (e.g., between 1 and 10000)
                                for i in range(8, len(chunk)-24, 8):
                                    val = struct.unpack('<d', chunk[i:i+8])[0]
                                    if 1 <= val <= 10000:
                                        # This might be a price value
                                        continue
                                    
                                # If we got here, it might be a valid record
                                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                                open_val = struct.unpack('<d', chunk[8:16])[0]
                                high_val = struct.unpack('<d', chunk[16:24])[0]
                                low_val = struct.unpack('<d', chunk[24:32])[0]
                                close_val = struct.unpack('<d', chunk[32:40])[0]
                                
                                # Volume might be in the next bytes
                                volume_val = 0
                                if pos+window_size+8 <= len(content):
                                    volume_val = struct.unpack('<d', content[pos+window_size:pos+window_size+8])[0]
                                
                                # Only add if the values look reasonable
                                if (open_val > 0 and high_val >= open_val and low_val <= open_val and 
                                    close_val > 0 and volume_val >= 0):
                                    records.append({
                                        'datetime': timestamp,
                                        'open': open_val,
                                        'high': high_val,
                                        'low': low_val,
                                        'close': close_val,
                                        'volume': volume_val
                                    })
                            except:
                                pass
                    except:
                        pass
                
                pos += step_size
            
            if records:
                print(f"Extracted {len(records)} potential OHLCV records from DBN file")
                df = pd.DataFrame(records)
                df = df.set_index('datetime')
                return df
        except Exception as e:
            print(f"Error parsing DBN content: {e}")
    
    print("Could not extract OHLCV data from DBN file")
    return None

def load_ohlcv_from_jsonl(jsonl_path, use_synthetic=False):
    """
    Loads OHLCV data from the given path. Supports multiple formats:
    - JSONL (standard format with 't', 'o', 'h', 'l', 'c', 'v' fields)
    - DBN (binary format)
    
    Parameters:
    -----------
    jsonl_path : str
        Path to the data file.
    use_synthetic : bool
        If True, always use synthetic data instead of parsing the file.
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with OHLCV data and datetime index.
    """
    # Check if we should just use synthetic data
    if use_synthetic:
        print("\n⚠️ Using synthetic data for development purposes.")
        synthetic_csv = 'data/synthetic_es_ohlcv.csv'
        
        # Check if we already have synthetic data
        if os.path.exists(synthetic_csv):
            print(f"Loading existing synthetic data from {synthetic_csv}")
            df = pd.read_csv(synthetic_csv, index_col=0, parse_dates=True)
            df.index.name = 'datetime'
            return df
        
        # Generate new synthetic data
        print("Generating new synthetic OHLCV data...")
        df = generate_synthetic_ohlcv(num_days=5, seed=42)
        
        # Save it for future use
        save_synthetic_data(df, synthetic_csv)
        
        # Add synthetic flag column
        df['is_synthetic'] = True
        
        return df
    
    print(f"Loading OHLCV data from: {jsonl_path}")
    
    # First, try to determine if this is a DBN file
    try:
        with open(jsonl_path, 'rb') as f:
            header = f.read(16)
            
        if header.startswith(b'DBN'):
            print(f"Detected DBN format file, header: {header}")
            
            # Try to parse the DBN file
            dbn_df = parse_dbn_file(jsonl_path)
            if dbn_df is not None and not dbn_df.empty:
                print(f"Successfully parsed DBN file, found {len(dbn_df)} records")
                return dbn_df
            else:
                print("Failed to parse DBN file, will try other formats")
    except Exception as e:
        print(f"Error examining file header: {e}")
    
    # Next, try to parse as JSONL
    try:
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
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
                except json.JSONDecodeError:
                    continue
        
        if data:
            print(f"Successfully parsed JSONL file, found {len(data)} records")
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            return df
        else:
            print("No valid JSONL records found")
    except Exception as e:
        print(f"Error parsing as JSONL: {e}")
    
    # If we get here, we need to generate synthetic data
    print("\n⚠️ WARNING: Could not parse the input file as DBN or JSONL")
    print("⚠️ Using synthetic data for development purposes.")
    
    # Use our synthetic data generator
    synthetic_csv = 'data/synthetic_es_ohlcv.csv'
    
    # Check if we already have synthetic data
    if os.path.exists(synthetic_csv):
        print(f"Loading existing synthetic data from {synthetic_csv}")
        df = pd.read_csv(synthetic_csv, index_col=0, parse_dates=True)
        df.index.name = 'datetime'
    else:
        # Generate new synthetic data
        print("Generating new synthetic OHLCV data...")
        df = generate_synthetic_ohlcv(num_days=5, seed=42)
        
        # Save it for future use
        save_synthetic_data(df, synthetic_csv)
    
    # Add synthetic flag column
    df['is_synthetic'] = True
    
    return df 