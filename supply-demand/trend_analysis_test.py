"""
Test script for trend length analysis in supply-demand index
"""

import pandas as pd
import numpy as np
from new_supply_demand_index_engine import NewSupplyDemandIndexEngine

def load_mt5_vol75_data(file_path: str = "mt5_vol_75_20250101_20250831.csv") -> pd.DataFrame:
    """Load MT5 Vol 75 position data from CSV file"""
    print(f"Loading MT5 Vol 75 data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        df['minutes'] = pd.to_datetime(df['minutes'])
        df = df.sort_values('minutes').reset_index(drop=True)
        
        print(f"Loaded {len(df):,} position records")
        print(f"Date range: {df['minutes'].min()} to {df['minutes'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def calculate_net_exposure_from_positions(df: pd.DataFrame) -> pd.Series:
    """Calculate net exposure from LONG/SHORT position data"""
    print("Calculating net exposure from position data...")
    
    # Group by minute and position type
    grouped = df.groupby(['minutes', 'position'])['volume_usd'].sum().unstack(fill_value=0)
    
    # Calculate net exposure (LONG - SHORT)
    if 'LONG' in grouped.columns and 'SHORT' in grouped.columns:
        net_exposure = grouped['LONG'] - grouped['SHORT']
    elif 'LONG' in grouped.columns:
        net_exposure = grouped['LONG']
    elif 'SHORT' in grouped.columns:
        net_exposure = -grouped['SHORT']
    else:
        raise ValueError("No LONG or SHORT positions found in data")
    
    print(f"Generated {len(net_exposure)} minute-level exposure points")
    print(f"Exposure range: {net_exposure.min():,.0f} to {net_exposure.max():,.0f}")
    
    return net_exposure

def main():
    """Main function to test trend analysis"""
    print("=" * 80)
    print("SUPPLY-DEMAND INDEX TREND ANALYSIS TEST")
    print("=" * 80)
    
    # Load data and get a sample period
    df = load_mt5_vol75_data()
    
    # Take first 7 days for testing
    start_date = df['minutes'].min()
    end_date = start_date + pd.Timedelta(days=7)
    
    period_data = df[
        (df['minutes'] >= start_date) & 
        (df['minutes'] < end_date)
    ].copy()
    
    print(f"\nUsing 7-day period: {start_date.date()} to {end_date.date()}")
    print(f"Period contains {len(period_data):,} records")
    
    # Calculate exposure
    exposure_series = calculate_net_exposure_from_positions(period_data)
    
    # Initialize engine
    engine = NewSupplyDemandIndexEngine(
        sigma=0.30,
        scale=150_000,
        k=0.40,
        smoothness_factor=2.0,
        noise_injection_level=0.01
    )
    
    print("\nGenerating supply-demand index simulation...")
    
    # Generate price path (minute-level)
    path_seconds, drift_path, smoothed_drift_path, probability_path = engine.generate_dynamic_exposure_path(
        exposure_series=exposure_series,
        random_seed=42,
        ma_window=12,
        seconds_per_minute=60
    )
    
    # Sample to minute-level
    path_minutes = path_seconds[::60]  # Every 60 seconds = 1 minute
    
    # Ensure correct length
    if len(path_minutes) > len(exposure_series) + 1:
        path_minutes = path_minutes[:len(exposure_series) + 1]
    
    print(f"Generated price path with {len(path_minutes)} points")
    print(f"Price range: {path_minutes.min():.2f} - {path_minutes.max():.2f}")
    
    # Analyze trends
    print("\n" + "=" * 80)
    print("RUNNING TREND ANALYSIS")
    print("=" * 80)
    
    # Analyze price trends
    trend_results = engine.analyze_trend_lengths(
        price_path=path_minutes,
        exposure_series=exposure_series,
        min_trend_length=5  # Minimum 5 minutes for a trend
    )
    
    # Analyze exposure trends
    exposure_trend_results = engine.analyze_exposure_trends(
        exposure_series=exposure_series,
        min_trend_length=5  # Minimum 5 minutes for a trend
    )
    
    # Print results
    engine.print_trend_analysis(trend_results)
    
    # Print exposure trend analysis
    print("\n" + "=" * 60)
    print("EXPOSURE TREND ANALYSIS")
    print("=" * 60)
    
    print(f"Total exposure trends identified: {exposure_trend_results['total_exposure_trends']}")
    print(f"Average exposure trend length: {exposure_trend_results['avg_exposure_trend_length']:.1f} minutes")
    print(f"Exposure trend coverage: {exposure_trend_results['exposure_trend_coverage']:.1%} of total time")
    
    print("\nPOSITIVE EXPOSURE TRENDS:")
    pos_exp = exposure_trend_results['positive_exposure_trends']
    print(f"  Count: {pos_exp['count']}")
    if pos_exp['count'] > 0:
        print(f"  Average length: {pos_exp['avg_length']:.1f} minutes")
        print(f"  Length range: {pos_exp['min_length']} - {pos_exp['max_length']} minutes")
    
    print("\nNEGATIVE EXPOSURE TRENDS:")
    neg_exp = exposure_trend_results['negative_exposure_trends']
    print(f"  Count: {neg_exp['count']}")
    if neg_exp['count'] > 0:
        print(f"  Average length: {neg_exp['avg_length']:.1f} minutes")
        print(f"  Length range: {neg_exp['min_length']} - {neg_exp['max_length']} minutes")
    
    # Compare price trends vs exposure trends
    print("\n" + "=" * 60)
    print("PRICE TRENDS vs EXPOSURE TRENDS COMPARISON")
    print("=" * 60)
    
    price_avg_length = trend_results['avg_trend_length']
    exposure_avg_length = exposure_trend_results['avg_exposure_trend_length']
    
    print(f"Average PRICE trend length: {price_avg_length:.1f} minutes")
    print(f"Average EXPOSURE trend length: {exposure_avg_length:.1f} minutes")
    print(f"Ratio (Exposure/Price): {exposure_avg_length/price_avg_length:.1f}x")
    
    if exposure_avg_length > price_avg_length * 2:
        print("✓ EXPOSURE trends are MUCH LONGER than price trends")
        print("  → Exposure changes less frequently than price direction")
    elif exposure_avg_length > price_avg_length * 1.5:
        print("~ EXPOSURE trends are LONGER than price trends")
        print("  → Exposure is more persistent than price direction")
    elif exposure_avg_length < price_avg_length * 0.5:
        print("! EXPOSURE trends are MUCH SHORTER than price trends")
        print("  → Exposure changes more frequently than price direction")
    else:
        print("≈ EXPOSURE and PRICE trends have similar lengths")
        print("  → Similar persistence in both exposure and price direction")
    
    # Additional detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED TREND STATISTICS")
    print("=" * 60)
    
    if trend_results['total_trends'] > 0:
        all_trends = trend_results['all_trends']
        
        # Analyze by exposure level
        positive_exposure_trends = [t for t in all_trends if t['avg_exposure'] > 0]
        negative_exposure_trends = [t for t in all_trends if t['avg_exposure'] < 0]
        
        print(f"\nTrends with POSITIVE exposure: {len(positive_exposure_trends)}")
        if positive_exposure_trends:
            up_count = len([t for t in positive_exposure_trends if t['direction'] == 1])
            down_count = len([t for t in positive_exposure_trends if t['direction'] == -1])
            sideways_count = len([t for t in positive_exposure_trends if t['direction'] == 0])
            
            print(f"  Up trends: {up_count} ({up_count/len(positive_exposure_trends)*100:.1f}%)")
            print(f"  Down trends: {down_count} ({down_count/len(positive_exposure_trends)*100:.1f}%)")
            print(f"  Sideways trends: {sideways_count} ({sideways_count/len(positive_exposure_trends)*100:.1f}%)")
        
        print(f"\nTrends with NEGATIVE exposure: {len(negative_exposure_trends)}")
        if negative_exposure_trends:
            up_count = len([t for t in negative_exposure_trends if t['direction'] == 1])
            down_count = len([t for t in negative_exposure_trends if t['direction'] == -1])
            sideways_count = len([t for t in negative_exposure_trends if t['direction'] == 0])
            
            print(f"  Up trends: {up_count} ({up_count/len(negative_exposure_trends)*100:.1f}%)")
            print(f"  Down trends: {down_count} ({down_count/len(negative_exposure_trends)*100:.1f}%)")
            print(f"  Sideways trends: {sideways_count} ({sideways_count/len(negative_exposure_trends)*100:.1f}%)")
        
        # Length distribution
        all_lengths = [t['length'] for t in all_trends]
        print(f"\nTrend length distribution:")
        print(f"  Mean: {np.mean(all_lengths):.1f} minutes")
        print(f"  Median: {np.median(all_lengths):.1f} minutes")
        print(f"  Std: {np.std(all_lengths):.1f} minutes")
        print(f"  Min: {np.min(all_lengths)} minutes")
        print(f"  Max: {np.max(all_lengths)} minutes")
        
        # Percentiles
        print(f"  25th percentile: {np.percentile(all_lengths, 25):.1f} minutes")
        print(f"  75th percentile: {np.percentile(all_lengths, 75):.1f} minutes")
        print(f"  90th percentile: {np.percentile(all_lengths, 90):.1f} minutes")
    
    print("\n" + "=" * 80)
    print("TREND ANALYSIS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
