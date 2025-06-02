import os
import pandas as pd
import databento as db
from datetime import datetime
import json
import subprocess
import warnings
import numpy as np
from src.synthetic_data import generate_synthetic_ohlcv

def decompress_zst_file(zst_path, output_path):
    """
    This function is kept for backward compatibility but is no longer needed
    since we'll use the Databento SDK directly.
    """
    print("Using Databento SDK to parse the DBN.ZST file directly.")
    return zst_path

def parse_dbn_with_databento(file_path):
    """
    Parse a DBN.ZST file using the Databento Python SDK.
    
    Parameters:
    -----------
    file_path : str
        Path to the DBN.ZST file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and OHLCV columns
    """
    print(f"Parsing DBN.ZST file: {file_path}")
    
    try:
        # Use databento.read_dbn to read the file directly to DataFrame
        dbn_store = db.read_dbn(file_path)
        df = dbn_store.to_df()
        
        # Print columns for debugging
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check if DataFrame already has a datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            print("DataFrame already has a datetime index")
            has_datetime_index = True
        else:
            has_datetime_index = False
            print(f"Index type: {type(df.index)}")
        
        # Extract required columns: open, high, low, close, volume/size
        # Adjust column names if needed based on actual file format
        required_columns = ['open', 'high', 'low', 'close']
        if 'size' in df.columns:
            volume_col = 'size'
        elif 'volume' in df.columns:
            volume_col = 'volume'
        else:
            raise ValueError("Missing required volume column. Neither 'size' nor 'volume' found.")
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # If we don't have a timestamp column and don't have a datetime index
        if 'ts_event' not in df.columns and not has_datetime_index:
            # Check if we have a timestamp column with a different name
            timestamp_alternatives = ['timestamp', 'time', 'date', 'datetime']
            timestamp_col = None
            
            for col in timestamp_alternatives:
                if col in df.columns:
                    print(f"Using '{col}' column as timestamp")
                    timestamp_col = col
                    break
                    
            if not timestamp_col:
                # We have no timestamp column at all, check if index might be a timestamp
                if df.index.name in timestamp_alternatives:
                    print(f"Using index as timestamp (name: {df.index.name})")
                    has_datetime_index = True
                else:
                    # Last resort - check if the symbol column has date information
                    if 'symbol' in df.columns:
                        print("Attempting to extract date from 'symbol' column...")
                        # Some DBN files have dates in the symbol names
                        # This is a fallback and might not always work
                        try:
                            # Extract date from first row's symbol
                            symbol = df['symbol'].iloc[0]
                            print(f"First symbol: {symbol}")
                            
                            # Try to parse date parts from the symbol (very implementation-specific)
                            # This assumes something like "ES_20240602" or similar format
                            import re
                            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', symbol)
                            if date_match:
                                year, month, day = date_match.groups()
                                base_date = pd.Timestamp(f"{year}-{month}-{day}")
                                print(f"Extracted base date: {base_date}")
                                
                                # Create timestamps starting from market open (9:30 AM)
                                market_open = pd.Timestamp(f"{year}-{month}-{day} 09:30:00")
                                minutes = pd.date_range(
                                    start=market_open,
                                    periods=len(df),
                                    freq='1min'
                                )
                                print(f"Created time range: {minutes[0]} to {minutes[-1]}")
                                
                                # Add as a new column
                                df['ts_event'] = minutes
                                timestamp_col = 'ts_event'
                            else:
                                raise ValueError("Could not extract date from symbol")
                        except Exception as e:
                            print(f"Error extracting date from symbol: {e}")
                            raise ValueError("No timestamp column found and couldn't create one")
        
        # Create a new DataFrame with required columns
        result_columns = {
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df[volume_col]
        }
        
        if has_datetime_index:
            # If we already have a datetime index, use it
            result_df = pd.DataFrame(result_columns, index=df.index)
        elif 'ts_event' in df.columns:
            # Use ts_event column as index
            result_df = pd.DataFrame(result_columns)
            result_df.index = pd.to_datetime(df['ts_event'], unit='ns')
        elif timestamp_col:
            # Use the alternative timestamp column
            result_df = pd.DataFrame(result_columns)
            result_df.index = pd.to_datetime(df[timestamp_col])
        else:
            # As a last resort, create a synthetic index based on 1-minute intervals
            print("WARNING: No timestamp column found. Creating synthetic timestamps.")
            start_time = pd.Timestamp('2024-06-03 09:30:00')  # Default to a reasonable date
            minutes = pd.date_range(start=start_time, periods=len(df), freq='1min')
            result_df = pd.DataFrame(result_columns, index=minutes)
        
        # Set index name
        result_df.index.name = 'datetime'
        
        # Sort by datetime
        result_df = result_df.sort_index()
        
        print(f"Successfully parsed {len(result_df)} records from {file_path}")
        print(f"Date range: {result_df.index.min()} to {result_df.index.max()}")
        return result_df
        
    except Exception as e:
        print(f"Error parsing DBN.ZST file: {e}")
        raise

def load_ohlcv_from_jsonl(file_path, use_synthetic=False):
    """
    Load OHLCV data from a JSONL file or generate synthetic data.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSONL file or DBN.ZST file
    use_synthetic : bool
        Whether to use synthetic data instead of parsing real data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and OHLCV columns
    """
    if use_synthetic:
        # Generate synthetic data instead of parsing
        print("Generating new synthetic OHLCV data...")
        df = generate_synthetic_ohlcv(start_date='2024-06-03', num_days=5, seed=42)
        
        # Save to CSV for inspection
        csv_path = 'data/synthetic_es_ohlcv.csv'
        df.to_csv(csv_path)
        print(f"Synthetic data saved to {csv_path}")
        
        return df
    
    # Check if the file exists and looks like a DBN.ZST file
    if file_path and os.path.exists(file_path) and file_path.endswith('.dbn.zst'):
        try:
            # Use Databento SDK to parse the DBN.ZST file
            return parse_dbn_with_databento(file_path)
        except Exception as e:
            print(f"Failed to parse DBN.ZST file with Databento SDK: {e}")
            raise  # Re-raise the exception to be handled by the caller
    
    # Check if it's a JSONL file (for backward compatibility)
    elif file_path and os.path.exists(file_path) and file_path.endswith('.jsonl'):
        try:
            # Try to parse the JSONL file (legacy method)
            print(f"Attempting to parse JSONL file: {file_path}")
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            if not data:
                print("No valid JSON lines found in file.")
                raise ValueError("No valid JSON lines found in file.")
            
            # Create DataFrame from parsed data
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime and set as index
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('datetime')
            
            # Ensure all required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Keep only the required columns
            df = df[required_cols]
            
            return df
            
        except Exception as e:
            print(f"Failed to parse JSONL file: {e}")
            raise  # Re-raise the exception to be handled by the caller
    else:
        # If file doesn't exist or is not a recognized format
        if file_path:
            msg = f"File not found or invalid format: {file_path}"
            print(msg)
            raise FileNotFoundError(msg)
        else:
            msg = "No file path provided."
            print(msg)
            raise ValueError(msg)

if __name__ == "__main__":
    # Path to the DBN.ZST file
    dbn_path = "data/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst"
    
    # Test the DBN parsing function
    if os.path.exists(dbn_path):
        print(f"Testing DBN parsing with file: {dbn_path}")
        try:
            df = parse_dbn_with_databento(dbn_path)
            print("\nSample data:")
            print(df.head())
            print(f"\nShape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        except Exception as e:
            print(f"Error testing DBN parsing: {e}")
            print("\nExiting due to parsing error.")
    else:
        print(f"DBN file not found: {dbn_path}")
        print("Exiting as no valid data file was found.") 