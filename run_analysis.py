from src.parse_zst_ohlcv import decompress_zst_file, load_ohlcv_from_jsonl, parse_dbn_with_databento
from src.volume_cluster import identify_volume_clusters, analyze_forward_returns
import os
import subprocess
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Volume Cluster Analyzer')
    parser.add_argument('--use-synthetic', action='store_true', 
                        help='Use synthetic data instead of parsing real data')
    parser.add_argument('--zst-path', type=str, 
                        default='data/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst',
                        help='Path to the ZST file')
    parser.add_argument('--output-path', type=str, 
                        default='data/output.jsonl',
                        help='Path to save the decompressed output')
    parser.add_argument('--volume-multiplier', type=float, default=3.0,
                        help='Multiplier for volume threshold (default: 3.0)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Paths
    zst_path = args.zst_path
    jsonl_path = args.output_path
    
    try:
        # Print header
        print("\n" + "="*80)
        print("VOLUME CLUSTER ANALYZER")
        print("="*80)
        
        # Step 1: Data Loading
        if args.use_synthetic:
            print("\nStep 1: Using synthetic data (skipping decompression)")
            df = load_ohlcv_from_jsonl(None, use_synthetic=True)
        else:
            # Attempt to load real ES futures data from the DBN.ZST file
            print("\nStep 1: Loading real ES futures data...")
            
            if not os.path.exists(zst_path):
                print(f"ERROR: ZST file not found at {zst_path}")
                print("Please ensure the ZST file is in the correct location.")
                print("Use --use-synthetic flag to use synthetic data instead.")
                sys.exit(f"File not found: {zst_path}")
            
            try:
                # Use Databento SDK to parse the DBN.ZST file directly
                df = parse_dbn_with_databento(zst_path)
                
                # Print information about the loaded data
                print(f"\nSuccessfully loaded real ES futures data:")
                print(f"Number of rows: {len(df):,}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Timespan: {(df.index.max() - df.index.min()).days + 1} days")
                
                # Save a backup copy of the parsed data
                backup_path = 'data/es_ohlcv_real.csv'
                df.to_csv(backup_path)
                print(f"Saved backup copy to {backup_path}")
                
            except Exception as e:
                print(f"\nERROR: Failed to load real market data: {e}")
                print("\nTraceback:")
                traceback.print_exc()
                sys.exit("Failed to load real market data. Please check the DBN parser.")
        
        # Verify if this is real or synthetic data
        is_synthetic = 'is_synthetic' in df.columns if not df.empty else False
        if is_synthetic:
            print("\n⚠️ Using synthetic data for development purposes.")
        else:
            print("\nUsing real market data for analysis.")
        
        print("\nData preview:")
        print(df.head())
        
        # Data shape and stats
        print(f"\nDataset shape: {df.shape}")
        print("\nDate range:")
        print(f"  Start: {df.index.min()}")
        print(f"  End:   {df.index.max()}")
        print(f"  Span:  {(df.index.max() - df.index.min()).days + 1} days")
        
        print("\nSummary statistics:")
        print(df.describe())
    
        # Step 3: Identify clusters on a day-by-day basis
        print("\nStep 3: Identifying volume clusters by day...")
        print(f"Using {args.volume_multiplier}x daily average 15-minute volume as threshold")
        clusters = identify_volume_clusters(df, volume_multiplier=args.volume_multiplier)
        
        # Display clusters
        print("\nCluster results:")
        if clusters.empty:
            print("No clusters found with the default threshold.")
        else:
            print(f"Found {len(clusters)} volume clusters across {len(clusters['date'].unique())} trading days")
            print("\nClusters preview:")
            print(clusters.head())
            
            # Calculate cluster statistics
            total_periods = sum(len(day_data.resample('15min')) for _, day_data in df.groupby(df.index.date))
            cluster_percentage = (len(clusters) / total_periods) * 100
            print(f"\nCluster frequency: {cluster_percentage:.2f}% of all 15-minute periods")
            
            # Perform directional bias analysis
            print("\nStep 3.5: Analyzing directional bias...")
            return_analysis = analyze_forward_returns(clusters)
            
            # Save return analysis summary to CSV
            if return_analysis:
                summary_data = []
                for return_col, stats in return_analysis.items():
                    summary_data.append({
                        'horizon': stats['horizon'],
                        'sample_size': stats['count'],
                        'mean_return_pct': stats['mean_return_pct'],
                        'std_return_pct': stats['std_return_pct'],
                        't_statistic': stats['t_stat'],
                        'p_value': stats['p_value'],
                        'is_significant': stats['is_significant'],
                        'direction': stats['direction']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_path = "data/cluster_return_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"\n📊 Return analysis summary saved to: {summary_path}")
            
            # Save results for inspection
            clusters.to_csv("data/volume_clusters.csv")
            print(f"\nVolume clusters saved to data/volume_clusters.csv")
        
        # Create visualizations by day
        if not clusters.empty:
            print("\nStep 4: Creating visualizations...")
            
            # Get unique dates
            unique_dates = sorted(clusters['date'].unique())
            print(f"Generating charts for {len(unique_dates)} days with clusters")
            
            for i, date in enumerate(unique_dates):
                # Filter data for this date
                date_clusters = clusters[clusters['date'] == date]
                date_str = date.strftime('%Y-%m-%d')
                
                # Get all data for this date
                day_data = df[df.index.date == date]
                
                # Skip if no data
                if day_data.empty:
                    continue
                    
                # Create a figure with price and volume subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                fig.suptitle(f'Volume Clusters for {date_str}', fontsize=16)
                
                # Resample to 15-minute intervals for this day
                df_15m = day_data.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                
                # Price subplot
                ax1.plot(df_15m.index, df_15m['close'], label='Close Price')
                
                # Mark cluster times on price chart
                for idx in date_clusters.index:
                    strength = date_clusters.loc[idx, 'cluster_strength']
                    label = f"{strength:.1f}x" if idx == date_clusters.index[0] else None
                    ax1.axvline(idx, color='r', alpha=min(0.9, strength/10), 
                               linestyle='--', label=label)
                
                ax1.set_title('Price with Volume Clusters')
                ax1.set_ylabel('Price')
                ax1.legend()
                
                # Format x-axis to show hours
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                
                # Volume subplot
                bar_width = 0.008  # Adjust for visual clarity
                ax2.bar(df_15m.index, df_15m['volume'], width=bar_width, color='b', alpha=0.6, label='Volume')
                
                # Highlight cluster volumes
                ax2.bar(date_clusters.index, date_clusters['volume'], width=bar_width, color='r', alpha=0.8, label='Cluster Volume')
                
                # Add threshold line
                if not date_clusters.empty:
                    threshold = date_clusters['threshold'].iloc[0]
                    avg = date_clusters['avg_volume'].iloc[0]
                    ax2.axhline(threshold, color='r', linestyle='--', 
                              label=f'{args.volume_multiplier}x Daily Avg ({threshold:.0f})')
                    ax2.axhline(avg, color='g', linestyle=':', 
                              label=f'Daily Avg ({avg:.0f})')
                
                ax2.set_ylabel('Volume')
                ax2.set_xlabel('Time')
                ax2.legend()
                
                # Save the figure
                plt.tight_layout()
                os.makedirs('data/daily_charts', exist_ok=True)
                plt.savefig(f'data/daily_charts/volume_clusters_{date_str}.png')
                plt.close()
                
                print(f"  Chart for {date_str} saved to data/daily_charts/volume_clusters_{date_str}.png")
            
            # Create summary heatmap of cluster strengths
            if len(unique_dates) > 1:
                print("\nCreating cluster strength heatmap...")
                
                # Prepare data for heatmap
                # Resample all data to 15-minute intervals
                all_15m = df.resample('15min').agg({'volume': 'sum'})
                
                # Create time bins (hours of the day)
                all_15m['hour'] = all_15m.index.hour + all_15m.index.minute/60
                
                # Create empty heatmap data
                hours = np.arange(0, 24, 0.25)  # 15-minute intervals
                dates = unique_dates
                heatmap_data = np.zeros((len(dates), len(hours)))
                
                # Fill heatmap with cluster strengths
                for i, date in enumerate(dates):
                    date_clusters = clusters[clusters['date'] == date]
                    for _, cluster in date_clusters.iterrows():
                        hour_idx = int((cluster.name.hour + cluster.name.minute/60) * 4)  # 4 intervals per hour
                        if 0 <= hour_idx < len(hours):
                            heatmap_data[i, hour_idx] = cluster['cluster_strength']
                
                # Plot heatmap
                plt.figure(figsize=(15, 10))
                plt.imshow(heatmap_data, aspect='auto', cmap='hot')
                plt.colorbar(label='Cluster Strength (Volume / Daily Avg)')
                
                # Set labels
                plt.yticks(range(len(dates)), [d.strftime('%Y-%m-%d') for d in dates])
                plt.xticks(range(0, len(hours), 4), [f"{int(h)}:00" for h in hours[::4]])
                
                plt.title('Volume Cluster Strength by Time of Day')
                plt.xlabel('Time of Day')
                plt.ylabel('Date')
                
                plt.tight_layout()
                plt.savefig('data/cluster_strength_heatmap.png')
                plt.close()
                
                print("Cluster strength heatmap saved to data/cluster_strength_heatmap.png")
        
        # Create a single combined visualization
        print("\nCreating combined volume analysis chart...")
        
        # Group data by day and calculate daily stats
        df['date'] = df.index.date
        daily_stats = df.groupby('date').agg({
            'volume': 'sum',
            'close': ['first', 'last', 'max', 'min']
        })
        daily_stats.columns = ['total_volume', 'open', 'close', 'high', 'low']
        daily_stats['day_range'] = daily_stats['high'] - daily_stats['low']
        daily_stats['day_change'] = daily_stats['close'] - daily_stats['open']
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Price chart (top)
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        ax1.plot(df_15m.index, df_15m['close'], label='Price')
        
        # Mark all clusters on price chart
        if not clusters.empty:
            for idx in clusters.index:
                strength = clusters.loc[idx, 'cluster_strength']
                # Color-code by strength
                color = 'red' if strength >= 5 else 'orange' if strength >= 4 else 'yellow'
                alpha = min(0.8, strength / 10)  # Cap at 0.8 alpha
                ax1.axvline(idx, color=color, alpha=alpha, linewidth=1)
        
        ax1.set_title('Price with Volume Clusters')
        ax1.set_ylabel('Price')
        
        # Volume chart (middle)
        df_15m_vol = df.resample('15min').agg({'volume': 'sum'})
        ax2.bar(df_15m_vol.index, df_15m_vol['volume'], width=0.01, color='blue', alpha=0.6, label='15m Volume')
        
        # Mark all clusters on volume chart
        if not clusters.empty:
            # Plot each cluster's volume
            for idx, row in clusters.iterrows():
                ax2.bar(idx, row['volume'], width=0.01, color='red', alpha=0.8)
        
        ax2.set_ylabel('Volume')
        
        # Cluster strength chart (bottom)
        if not clusters.empty:
            # Create a scatter plot of cluster strengths
            scatter = ax3.scatter(
                clusters.index, 
                clusters['cluster_strength'],
                c=clusters['cluster_strength'],
                cmap='hot',
                s=clusters['cluster_strength'] * 10,  # Size based on strength
                alpha=0.7
            )
            
            # Add a colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Cluster Strength')
            
        ax3.set_ylabel('Cluster Strength')
        ax3.set_xlabel('Time')
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/combined_analysis.png')
        plt.close()
        
        print("Combined analysis chart saved to data/combined_analysis.png")
        
        # Create a summary table of clusters
        if not clusters.empty:
            print("\nCreating cluster summary table...")
            
            # Group clusters by date
            clusters_by_date = clusters.groupby('date')
            
            # Create summary DataFrame
            summary = []
            for date, group in clusters_by_date:
                date_str = date.strftime('%Y-%m-%d')
                
                # Get daily stats
                if date in daily_stats.index:
                    day_stats = daily_stats.loc[date]
                    
                    # Calculate cluster stats
                    num_clusters = len(group)
                    max_strength = group['cluster_strength'].max()
                    avg_strength = group['cluster_strength'].mean()
                    
                    # Get time of strongest cluster
                    strongest_idx = group['cluster_strength'].idxmax()
                    strongest_time = group.loc[strongest_idx].name.strftime('%H:%M')
                    
                    summary.append({
                        'date': date_str,
                        'num_clusters': num_clusters,
                        'max_strength': max_strength,
                        'avg_strength': avg_strength,
                        'strongest_time': strongest_time,
                        'daily_volume': day_stats['total_volume'],
                        'price_change': day_stats['day_change'],
                        'price_range': day_stats['day_range']
                    })
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv('data/cluster_summary.csv', index=False)
            
            print("Cluster summary saved to data/cluster_summary.csv")
            
        print("\nAnalysis complete! Check the 'data' directory for results.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(f"Analysis failed: {e}")

if __name__ == "__main__":
    main() 