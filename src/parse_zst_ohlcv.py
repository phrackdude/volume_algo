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
        
        # DEBUG: Print first few rows of raw data to verify column assignments
        print("\n=== RAW DATA INSPECTION ===")
        print("First 5 rows of raw DataFrame:")
        print(df.head())
        print(f"\nDataFrame shape: {df.shape}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        
        # DEBUG: Check for any abnormally low values in price columns
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            print("\n=== PRICE COLUMN ANALYSIS ===")
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                col_data = df[col]
                print(f"\n{col.upper()} column stats:")
                print(f"  Min: {col_data.min():.2f}")
                print(f"  Max: {col_data.max():.2f}")
                print(f"  Mean: {col_data.mean():.2f}")
                print(f"  Median: {col_data.median():.2f}")
                
                # Check for abnormally low values (< 100 for ES futures)
                low_values = col_data[col_data < 100]
                if len(low_values) > 0:
                    print(f"  ⚠️  WARNING: Found {len(low_values)} values < 100!")
                    print(f"  Sample low values: {low_values.head(10).tolist()}")
                
                # Sample of first 10 values
                print(f"  First 10 values: {col_data.head(10).tolist()}")
            
            # Check OHLC relationship validity
            print("\n=== OHLC RELATIONSHIP VALIDATION ===")
            invalid_ohlc = df[(df['low'] > df['high']) | 
                             (df['open'] < df['low']) | 
                             (df['open'] > df['high']) |
                             (df['close'] < df['low']) | 
                             (df['close'] > df['high'])]
            
            if len(invalid_ohlc) > 0:
                print(f"⚠️  WARNING: Found {len(invalid_ohlc)} rows with invalid OHLC relationships!")
                print("Sample invalid rows:")
                print(invalid_ohlc[['open', 'high', 'low', 'close']].head())
            else:
                print("✅ All OHLC relationships are valid")
        
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
        
        # DEBUG: Verify column assignment order matches expected schema
        print(f"\n=== COLUMN ASSIGNMENT VERIFICATION ===")
        print(f"Expected order: ['ts_event', 'open', 'high', 'low', 'close', 'volume']")
        print(f"Actual columns: {df.columns.tolist()}")
        print(f"Volume column being used: '{volume_col}'")
        
        # FILTER: Remove spread contracts to keep only outright futures contracts
        if 'symbol' in df.columns:
            print(f"\n=== FILTERING SPREAD CONTRACTS ===")
            original_count = len(df)
            
            # Identify spread contracts (contain hyphen)
            spread_mask = df['symbol'].str.contains('-', na=False)
            spread_count = spread_mask.sum()
            
            # Keep only outright contracts (no hyphen in symbol)
            df = df[~spread_mask].copy()
            
            print(f"Original records: {original_count:,}")
            print(f"Spread contracts removed: {spread_count:,}")
            print(f"Outright contracts kept: {len(df):,}")
            
            # Verify remaining symbols
            remaining_symbols = df['symbol'].unique()
            print(f"Remaining symbols: {remaining_symbols}")
            
            # Verify price ranges after filtering
            if len(df) > 0:
                print(f"Price range after filtering:")
                print(f"  Open: {df['open'].min():.2f} - {df['open'].max():.2f}")
                print(f"  High: {df['high'].min():.2f} - {df['high'].max():.2f}")
                print(f"  Low: {df['low'].min():.2f} - {df['low'].max():.2f}")
                print(f"  Close: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
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
        
        # DEBUG: Print sample of result columns before creating DataFrame
        print(f"\n=== RESULT COLUMNS SAMPLE ===")
        for col_name, col_data in result_columns.items():
            print(f"{col_name}: {col_data.head(3).tolist()}")
        
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
        
        # DEBUG: Final validation of result DataFrame
        print(f"\n=== FINAL RESULT VALIDATION ===")
        print(f"Final DataFrame shape: {result_df.shape}")
        print("Final DataFrame sample:")
        print(result_df.head())
        
        # Check for any remaining low price anomalies
        low_anomalies = result_df[result_df['low'] < 100]
        if len(low_anomalies) > 0:
            print(f"\n⚠️  WARNING: Final DataFrame still contains {len(low_anomalies)} rows with low < 100!")
            print("Sample anomalies:")
            print(low_anomalies.head())
        else:
            print("\n✅ Final DataFrame has no low price anomalies")
        
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