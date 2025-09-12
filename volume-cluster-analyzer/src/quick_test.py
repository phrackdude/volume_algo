"""
Quick test to verify V6 data loading and basic robustness analysis
"""

import pandas as pd
import numpy as np

def quick_test():
    """Quick test of V6 data and basic sensitivity analysis"""
    
    print("🔬 QUICK V6 ROBUSTNESS TEST")
    print("=" * 40)
    
    try:
        # Load V6 results
        print("Loading V6 data...")
        df = pd.read_csv("../data/backtest_results_v6.csv")
        print(f"✅ Loaded {len(df)} trades successfully")
        
        # Basic performance metrics
        baseline_return = df['position_adjusted_return'].mean()
        win_rate = (df['net_return'] > 0).mean()
        avg_multiplier = df['bayesian_multiplier'].mean()
        
        print(f"\n📊 V6 BASE PERFORMANCE:")
        print(f"   Position-adjusted return: {baseline_return:.4%}")
        print(f"   Win rate: {win_rate:.2%}")
        print(f"   Average Bayesian multiplier: {avg_multiplier:.3f}")
        
        # Test basic parameter sensitivity
        print(f"\n🎯 QUICK PARAMETER SENSITIVITY TEST:")
        
        # Test SCALING_FACTOR sensitivity
        scaling_factors = [4.5, 6.0, 7.5]
        for sf in scaling_factors:
            # Simulate new multipliers
            new_multipliers = []
            for _, row in df.iterrows():
                expected_p = row['bayesian_expected_p']
                total_trades = row['bayesian_total_trades']
                
                if total_trades >= 3 and expected_p > 0.5:
                    raw_mult = 1.0 + (expected_p - 0.5) * sf
                    new_mult = min(raw_mult, 3.0)
                else:
                    new_mult = 1.0
                
                new_multipliers.append(new_mult)
            
            # Calculate new returns
            new_returns = df['net_return'] * new_multipliers
            new_avg_return = new_returns.mean()
            change = (new_avg_return / baseline_return - 1) * 100
            
            print(f"   SCALING_FACTOR {sf}: {new_avg_return:.4%} ({change:+.1f}%)")
        
        # Test MAX_MULTIPLIER sensitivity
        print(f"\n🎯 MAX_MULTIPLIER SENSITIVITY:")
        max_multipliers = [2.5, 3.0, 3.5]
        for mm in max_multipliers:
            # Cap multipliers at new maximum
            capped_multipliers = df['bayesian_multiplier'].apply(lambda x: min(x, mm))
            capped_returns = df['net_return'] * capped_multipliers
            capped_avg_return = capped_returns.mean()
            change = (capped_avg_return / baseline_return - 1) * 100
            
            print(f"   MAX_MULTIPLIER {mm}: {capped_avg_return:.4%} ({change:+.1f}%)")
        
        # Basic transaction cost stress test
        print(f"\n💰 QUICK TRANSACTION COST TEST:")
        cost_scenarios = [
            ("Low Cost", 1.00, 0.5),
            ("Current", 2.50, 0.75),
            ("High Cost", 4.00, 1.0),
            ("Stress", 5.00, 2.0)
        ]
        
        for name, commission, slippage_ticks in cost_scenarios:
            # Recalculate costs
            new_costs = []
            for _, row in df.iterrows():
                pos_mult = row['position_multiplier']
                comm_cost = commission * pos_mult
                slip_cost = slippage_ticks * 12.50 * pos_mult
                total_cost = (comm_cost + slip_cost) / (5000 * pos_mult)
                new_costs.append(total_cost)
            
            new_net_returns = df['gross_return'] - new_costs
            new_pos_returns = new_net_returns * df['bayesian_multiplier']
            avg_return = new_pos_returns.mean()
            change = (avg_return / baseline_return - 1) * 100
            
            print(f"   {name}: {avg_return:.4%} ({change:+.1f}%)")
        
        # Robustness assessment
        print(f"\n🎯 QUICK ROBUSTNESS ASSESSMENT:")
        
        # Check Bayesian utilization
        bayesian_trades = len(df[df['bayesian_diagnostic_method'] == 'bayesian'])
        utilization = bayesian_trades / len(df)
        print(f"   Bayesian utilization: {utilization:.1%}")
        
        # Check multiplier distribution
        print(f"   Multiplier stats: min={df['bayesian_multiplier'].min():.2f}, max={df['bayesian_multiplier'].max():.2f}")
        
        # Overall assessment
        print(f"\n🎯 QUICK ASSESSMENT RESULT:")
        if utilization > 0.9 and win_rate > 0.6:
            print(f"   ✅ ROBUST: High utilization and win rate")
        elif utilization > 0.8 and win_rate > 0.55:
            print(f"   ⚠️  MODERATE: Good but monitor closely")
        else:
            print(f"   ❌ NEEDS REVIEW: Low performance metrics")
        
        print(f"\n   📋 RECOMMENDATION: Proceed to full robustness testing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 