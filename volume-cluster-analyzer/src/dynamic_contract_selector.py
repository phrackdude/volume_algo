#!/usr/bin/env python3
"""
Dynamic ES Futures Contract Selector
Selects the most liquid ES contract for daily trading based on volume analysis
Ensures optimal execution and liquidity for the V6 Bayesian strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_contract_liquidity(market_data_dict):
    """
    Analyze multiple ES contracts and return the most liquid one
    
    Args:
        market_data_dict: Dictionary with contract symbols as keys and volume data as values
        Example: {
            'ES JUN25': {'volume': 1225850, 'bid': 6009.50, 'ask': 6010.00},
            'ES SEP25': {'volume': 20693, 'bid': 6048.00, 'ask': 6069.00}
        }
    
    Returns:
        dict: Best contract with analysis
    """
    
    print("🔍 ES FUTURES LIQUIDITY ANALYSIS")
    print("=" * 40)
    
    contract_analysis = []
    
    for contract, data in market_data_dict.items():
        volume = data.get('volume', 0)
        bid = data.get('bid', 0)
        ask = data.get('ask', 0)
        
        # Calculate spread in ticks (ES tick = 0.25)
        spread_ticks = (ask - bid) / 0.25 if ask > bid else 999
        
        # Calculate liquidity score
        # Higher volume = better, lower spread = better
        if volume > 0 and spread_ticks < 10:
            liquidity_score = volume / (spread_ticks ** 2)
        else:
            liquidity_score = 0
        
        contract_analysis.append({
            'contract': contract,
            'volume': volume,
            'bid': bid,
            'ask': ask,
            'spread_ticks': spread_ticks,
            'liquidity_score': liquidity_score
        })
        
        print(f"{contract}:")
        print(f"  Volume: {volume:,}")
        print(f"  Spread: {spread_ticks:.1f} ticks")
        print(f"  Liquidity Score: {liquidity_score:,.0f}")
        print()
    
    # Sort by liquidity score (highest first)
    contract_analysis.sort(key=lambda x: x['liquidity_score'], reverse=True)
    
    best_contract = contract_analysis[0]
    
    print("🎯 RECOMMENDED CONTRACT FOR TODAY:")
    print(f"   {best_contract['contract']}")
    print(f"   Volume: {best_contract['volume']:,}")
    print(f"   Spread: {best_contract['spread_ticks']:.1f} ticks")
    
    # Safe liquidity advantage calculation
    if len(contract_analysis) > 1 and contract_analysis[1]['liquidity_score'] > 0:
        advantage = best_contract['liquidity_score'] / contract_analysis[1]['liquidity_score']
        print(f"   Liquidity Advantage: {advantage:.1f}x better than next best")
    else:
        print(f"   Liquidity Advantage: DOMINANT (others have zero liquidity)")
    
    return best_contract

def get_contract_expiration_dates():
    """
    Return approximate expiration dates for ES futures contracts
    ES futures expire on the 3rd Friday of the contract month
    """
    return {
        'ES JUN25': datetime(2025, 6, 20),  # 3rd Friday of June 2025
        'ES SEP25': datetime(2025, 9, 19),  # 3rd Friday of September 2025
        'ES DEC25': datetime(2025, 12, 19), # 3rd Friday of December 2025
        'ES MAR26': datetime(2026, 3, 20),  # 3rd Friday of March 2026
    }

def check_roll_period(contract_name, current_date=None):
    """
    Check if we're in the roll period (2 weeks before expiration)
    During roll periods, volume often shifts to the next contract
    """
    if current_date is None:
        current_date = datetime.now()
    
    expiration_dates = get_contract_expiration_dates()
    
    if contract_name in expiration_dates:
        expiration = expiration_dates[contract_name]
        days_to_expiration = (expiration - current_date).days
        
        if days_to_expiration <= 14:
            return True, days_to_expiration
    
    return False, None

def daily_contract_recommendation():
    """
    Daily contract selection recommendation with roll period warnings
    """
    print("📅 DAILY ES CONTRACT RECOMMENDATION")
    print("=" * 45)
    
    current_date = datetime.now()
    print(f"Date: {current_date.strftime('%Y-%m-%d')}")
    print()
    
    # Check roll periods for all contracts
    expiration_dates = get_contract_expiration_dates()
    
    for contract, expiration in expiration_dates.items():
        is_roll_period, days_left = check_roll_period(contract, current_date)
        
        if is_roll_period:
            print(f"⚠️  {contract} ROLL PERIOD WARNING:")
            print(f"   {days_left} days until expiration")
            print(f"   Monitor volume migration to next contract")
            print()
    
    print("📋 DAILY CHECKLIST:")
    print("1. Check real-time volume for all ES contracts")
    print("2. Calculate spreads (bid-ask)")
    print("3. Select highest liquidity score contract")
    print("4. Configure trading platform")
    print("5. Run V6 Bayesian strategy")
    print()
    
    print("💡 PRO TIP:")
    print("   Always use the contract with >80% of total ES volume")
    print("   Avoid contracts with >2 tick spreads")
    print("   Switch contracts during roll periods when volume shifts")

def example_usage():
    """
    Example of how to use the contract selector with your current data
    """
    print("\n" + "="*60)
    print("EXAMPLE: Using Your Current Market Data")
    print("="*60)
    
    # Your current market data from the screenshot
    current_market_data = {
        'ES JUN25': {
            'volume': 1225850,
            'bid': 6009.50,
            'ask': 6010.00
        },
        'ES SEP25': {
            'volume': 20693,
            'bid': 6048.00,
            'ask': 6069.00  # Note: Large spread!
        },
        'ES DEC25': {
            'volume': 185,
            'bid': 6950.00,
            'ask': 6150.00  # Note: Crossed market - likely stale data
        }
    }
    
    best_contract = analyze_contract_liquidity(current_market_data)
    
    print("\n🎯 CONCLUSION:")
    print(f"   Trade {best_contract['contract']} today")
    print(f"   Your V6 strategy will have optimal execution")

if __name__ == "__main__":
    daily_contract_recommendation()
    example_usage() 