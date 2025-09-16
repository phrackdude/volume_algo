#!/usr/bin/env python3
"""
Volume Analysis Test Script
Tests volume cluster detection and analysis using real Databento data
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from databento_connector import DatabentoConnector
from config import config, load_config_from_file
from volume_cluster import identify_volume_clusters

class VolumeAnalysisTester:
    """Test volume cluster detection and analysis"""
    
    def __init__(self):
        # Load configuration
        load_config_from_file()
        
        # Set the API key from config
        os.environ['DATABENTO_API_KEY'] = config.databento_api_key
        
        self.databento_connector = DatabentoConnector(config.databento_api_key)
        
        # Volume analysis parameters
        self.VOLUME_THRESHOLD = 4.0  # Same as backtest
        self.MIN_VOLUME_RATIO = 60.0  # From backtest
        
    def _process_databento_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Databento data format for analysis"""
        try:
            print(f"📊 Raw Databento data columns: {list(df.columns)}")
            print(f"📊 Raw Databento data shape: {df.shape}")
            
            # Handle different column formats
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in df.columns]
            
            if len(available_columns) < 5:
                print(f"⚠️  Missing required columns. Available: {list(df.columns)}")
                # Try to map common column names
                column_mapping = {
                    'ts_event': 'timestamp',
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }
                
                # Rename columns if they exist
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
            
            # Select only the columns we need
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in required_columns):
                print(f"❌ Missing required columns. Available: {list(df.columns)}")
                raise ValueError("Missing required OHLCV columns")
            
            # Select only OHLCV columns
            df = df[required_columns]
            
            print(f"✅ Processed data: {len(df)} records")
            print(f"📊 Data range: {df.index.min()} to {df.index.max()}")
            print(f"📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            print(f"📊 Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing Databento data: {e}")
            raise
    
    async def get_historical_data(self, start_time: datetime, end_time: datetime, contract: str = "ES JUN25"):
        """Get and process historical data"""
        print(f"📈 Fetching historical data: {contract} from {start_time} to {end_time}")
        
        # Initialize Databento connector
        initialized = await self.databento_connector.initialize()
        if not initialized:
            print("❌ Failed to initialize Databento connector")
            return pd.DataFrame()
        
        # Adjust end_time to be within available data range
        adjusted_end_time = min(end_time, datetime.now() - timedelta(hours=3))
        print(f"📈 Adjusted end time to: {adjusted_end_time}")
        
        try:
            historical_data = await self.databento_connector.get_historical_data(
                contract, start_time, adjusted_end_time
            )
            
            if historical_data.empty:
                print("⚠️  No historical data available")
                return pd.DataFrame()
            
            # Process the data
            processed_data = self._process_databento_data(historical_data)
            return processed_data
            
        except Exception as e:
            print(f"❌ Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def analyze_volume_patterns(self, df: pd.DataFrame):
        """Analyze volume patterns in the data"""
        print("\n" + "="*60)
        print("📊 VOLUME PATTERN ANALYSIS")
        print("="*60)
        
        # Basic volume statistics
        print(f"Total records: {len(df):,}")
        print(f"Volume statistics:")
        print(f"  Mean: {df['volume'].mean():,.0f}")
        print(f"  Median: {df['volume'].median():,.0f}")
        print(f"  Std: {df['volume'].std():,.0f}")
        print(f"  Min: {df['volume'].min():,}")
        print(f"  Max: {df['volume'].max():,}")
        
        # Volume by time of day
        df['hour'] = df.index.hour
        hourly_volume = df.groupby('hour')['volume'].agg(['mean', 'std', 'count'])
        
        print(f"\n📈 Hourly Volume Analysis:")
        print(f"{'Hour':<4} {'Mean Vol':<10} {'Std Vol':<10} {'Count':<8} {'Vol/Mean':<10}")
        print("-" * 50)
        
        overall_mean = df['volume'].mean()
        for hour in sorted(hourly_volume.index):
            stats = hourly_volume.loc[hour]
            vol_ratio = stats['mean'] / overall_mean
            print(f"{hour:<4} {stats['mean']:<10,.0f} {stats['std']:<10,.0f} {stats['count']:<8} {vol_ratio:<10.2f}")
        
        # Find high volume periods
        high_vol_threshold = df['volume'].quantile(0.95)  # Top 5%
        high_vol_periods = df[df['volume'] > high_vol_threshold]
        
        print(f"\n🔥 High Volume Periods (top 5%):")
        print(f"  Threshold: {high_vol_threshold:,.0f}")
        print(f"  Count: {len(high_vol_periods)}")
        print(f"  Percentage: {len(high_vol_periods)/len(df)*100:.1f}%")
        
        if len(high_vol_periods) > 0:
            print(f"  Time range: {high_vol_periods.index.min()} to {high_vol_periods.index.max()}")
            print(f"  Volume range: {high_vol_periods['volume'].min():,} - {high_vol_periods['volume'].max():,}")
    
    def test_volume_cluster_detection(self, df: pd.DataFrame):
        """Test the volume cluster detection logic"""
        print("\n" + "="*60)
        print("🔍 VOLUME CLUSTER DETECTION TEST")
        print("="*60)
        
        # Test with different thresholds
        thresholds = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        print(f"Testing volume cluster detection with different thresholds:")
        print(f"{'Threshold':<10} {'Clusters':<8} {'Avg Strength':<12} {'Max Strength':<12}")
        print("-" * 50)
        
        results = {}
        for threshold in thresholds:
            try:
                clusters = identify_volume_clusters(df, volume_multiplier=threshold)
                
                if not clusters.empty:
                    avg_strength = clusters['cluster_strength'].mean()
                    max_strength = clusters['cluster_strength'].max()
                    results[threshold] = {
                        'count': len(clusters),
                        'avg_strength': avg_strength,
                        'max_strength': max_strength,
                        'clusters': clusters
                    }
                    print(f"{threshold:<10} {len(clusters):<8} {avg_strength:<12.2f} {max_strength:<12.2f}")
                else:
                    results[threshold] = {'count': 0, 'avg_strength': 0, 'max_strength': 0, 'clusters': pd.DataFrame()}
                    print(f"{threshold:<10} {0:<8} {0:<12.2f} {0:<12.2f}")
                    
            except Exception as e:
                print(f"{threshold:<10} ERROR: {e}")
                results[threshold] = {'count': 0, 'error': str(e)}
        
        # Use the default threshold (4.0) for detailed analysis
        if 4.0 in results and results[4.0]['count'] > 0:
            clusters_4x = results[4.0]['clusters']
            print(f"\n📊 Detailed Analysis (4.0x threshold):")
            print(f"  Total clusters: {len(clusters_4x)}")
            print(f"  Average strength: {clusters_4x['cluster_strength'].mean():.2f}")
            print(f"  Max strength: {clusters_4x['cluster_strength'].max():.2f}")
            print(f"  Min strength: {clusters_4x['cluster_strength'].min():.2f}")
            
            # Calculate the ACTUAL volume ratios used in trading (like backtest does)
            daily_avg_1min = df['volume'].mean()
            print(f"\n🎯 TRADING VOLUME RATIO ANALYSIS:")
            print(f"  Daily 1-minute average volume: {daily_avg_1min:,.0f}")
            print(f"  Cluster volumes (15-min): {clusters_4x['volume'].min():,.0f} - {clusters_4x['volume'].max():,.0f}")
            
            # Calculate actual trading volume ratios
            trading_volume_ratios = clusters_4x['volume'] / daily_avg_1min
            print(f"  Trading volume ratios: {trading_volume_ratios.min():.1f}x - {trading_volume_ratios.max():.1f}x")
            print(f"  Average trading ratio: {trading_volume_ratios.mean():.1f}x")
            
            # Show top 5 clusters with trading ratios
            top_clusters = clusters_4x.nlargest(5, 'cluster_strength')
            print(f"\n🔥 Top 5 Volume Clusters (with Trading Ratios):")
            print(f"{'Time':<20} {'15min Vol':<12} {'Trading Ratio':<12} {'Strength':<10}")
            print("-" * 70)
            for idx, row in top_clusters.iterrows():
                trading_ratio = row['volume'] / daily_avg_1min
                print(f"{idx.strftime('%Y-%m-%d %H:%M'):<20} {row['volume']:<12,.0f} {trading_ratio:<12.1f}x {row['cluster_strength']:<10.2f}")
            
            return clusters_4x
        else:
            print(f"\n❌ No clusters found with 4.0x threshold")
            return pd.DataFrame()
    
    def analyze_cluster_timing(self, clusters_df: pd.DataFrame):
        """Analyze when clusters occur"""
        if clusters_df.empty:
            print("\n❌ No clusters to analyze")
            return
        
        print("\n" + "="*60)
        print("⏰ CLUSTER TIMING ANALYSIS")
        print("="*60)
        
        # Extract time components
        clusters_df['hour'] = clusters_df.index.hour
        clusters_df['minute'] = clusters_df.index.minute
        
        # Hourly distribution
        hourly_dist = clusters_df['hour'].value_counts().sort_index()
        print(f"Hourly distribution:")
        for hour, count in hourly_dist.items():
            print(f"  {hour:02d}:00 - {count} clusters")
        
        # Time of day analysis
        morning = clusters_df[(clusters_df['hour'] >= 9) & (clusters_df['hour'] < 12)]
        afternoon = clusters_df[(clusters_df['hour'] >= 12) & (clusters_df['hour'] < 16)]
        close = clusters_df[clusters_df['hour'] >= 16]
        
        print(f"\nTime of day distribution:")
        print(f"  Morning (9-12): {len(morning)} clusters ({len(morning)/len(clusters_df)*100:.1f}%)")
        print(f"  Afternoon (12-16): {len(afternoon)} clusters ({len(afternoon)/len(clusters_df)*100:.1f}%)")
        print(f"  Close (16+): {len(close)} clusters ({len(close)/len(clusters_df)*100:.1f}%)")
        
        # Cluster strength by time
        if len(clusters_df) > 0:
            print(f"\nCluster strength by time of day:")
            print(f"  Morning avg strength: {morning['cluster_strength'].mean():.2f}")
            print(f"  Afternoon avg strength: {afternoon['cluster_strength'].mean():.2f}")
            print(f"  Close avg strength: {close['cluster_strength'].mean():.2f}")
    
    def create_volume_visualization(self, df: pd.DataFrame, clusters_df: pd.DataFrame = None):
        """Create volume visualization"""
        print("\n📊 Creating volume visualization...")
        
        # Resample to 15-minute intervals for visualization
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Volume Analysis - ES Futures', fontsize=16)
        
        # Price chart
        ax1.plot(df_15m.index, df_15m['close'], label='Close Price', linewidth=1)
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Chart')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(df_15m.index, df_15m['volume'], width=0.01, alpha=0.7, label='15-min Volume')
        
        # Highlight clusters if available
        if clusters_df is not None and not clusters_df.empty:
            for idx, row in clusters_df.iterrows():
                ax2.bar(idx, row['volume'], width=0.01, color='red', alpha=0.8)
                ax1.axvline(idx, color='red', alpha=0.5, linestyle='--')
        
        # Add volume threshold line
        avg_volume = df_15m['volume'].mean()
        threshold = avg_volume * 4.0
        ax2.axhline(threshold, color='red', linestyle='--', label=f'4x Threshold ({threshold:,.0f})')
        ax2.axhline(avg_volume, color='green', linestyle=':', label=f'Avg Volume ({avg_volume:,.0f})')
        
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        ax2.set_title('Volume Chart with Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Save the plot
        os.makedirs('data', exist_ok=True)
        plot_path = 'data/volume_analysis_test.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Volume visualization saved to: {plot_path}")
    
    async def run_analysis(self, hours_back: int = 24):
        """Run the complete volume analysis"""
        print("🔍 VOLUME ANALYSIS TEST")
        print("=" * 50)
        
        # Get time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        print(f"📅 Analyzing volume data from {start_time} to {end_time}")
        
        # Get historical data
        df = await self.get_historical_data(start_time, end_time)
        
        if df.empty:
            print("❌ No data available for analysis")
            return
        
        # Analyze volume patterns
        self.analyze_volume_patterns(df)
        
        # Test volume cluster detection
        clusters_df = self.test_volume_cluster_detection(df)
        
        # Analyze cluster timing
        if not clusters_df.empty:
            self.analyze_cluster_timing(clusters_df)
        
        # Create visualization
        self.create_volume_visualization(df, clusters_df)
        
        print(f"\n✅ Volume analysis complete!")
        print(f"📊 Data: {len(df):,} records")
        print(f"🔥 Clusters: {len(clusters_df) if not clusters_df.empty else 0}")
        
        return df, clusters_df

async def main():
    """Main entry point"""
    print("🔍 VOLUME ANALYSIS TEST SCRIPT")
    print("=" * 50)
    
    # Initialize tester
    tester = VolumeAnalysisTester()
    
    # Get user input for time period
    print("\n📅 Select analysis period:")
    print("1. Last 6 hours")
    print("2. Last 12 hours") 
    print("3. Last 24 hours")
    print("4. Last 48 hours")
    print("5. Custom hours")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        hours_back = 6
    elif choice == "2":
        hours_back = 12
    elif choice == "3":
        hours_back = 24
    elif choice == "4":
        hours_back = 48
    elif choice == "5":
        try:
            hours_back = int(input("Enter hours to look back: "))
        except ValueError:
            print("Invalid input, using 24 hours as default")
            hours_back = 24
    else:
        print("Invalid choice, using 24 hours as default")
        hours_back = 24
    
    # Run analysis
    try:
        df, clusters_df = await tester.run_analysis(hours_back)
        
        if not clusters_df.empty:
            print(f"\n📋 CLUSTER SUMMARY:")
            print(f"  Total clusters found: {len(clusters_df)}")
            print(f"  Average cluster strength: {clusters_df['cluster_strength'].mean():.2f}")
            print(f"  Strongest cluster: {clusters_df['cluster_strength'].max():.2f}x average volume")
            
            # Show if any clusters meet the trading criteria
            trading_clusters = clusters_df[clusters_df['cluster_strength'] >= 4.0]
            print(f"  Clusters meeting 4.0x threshold: {len(trading_clusters)}")
            
            if len(trading_clusters) > 0:
                print(f"  These clusters would trigger trading signals!")
            else:
                print(f"  No clusters meet the trading threshold (4.0x average volume)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
