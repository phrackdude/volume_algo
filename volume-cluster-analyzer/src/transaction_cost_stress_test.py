"""
V6 Bayesian Transaction Cost Stress Testing
Tests strategy profitability under various transaction cost scenarios
to ensure robustness in different market conditions and broker setups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def stress_test_transaction_costs(results_file="./data/backtest_results_v6.csv"):
    """
    Test strategy performance under various transaction cost scenarios
    """
    
    df = pd.read_csv(results_file)
    
    # Base transaction costs (from V6)
    base_commission = 2.50  # per round trip
    base_slippage = 0.75    # ticks
    tick_value = 12.50
    
    # Stress test scenarios
    cost_scenarios = {
        'Low Cost Broker': {'commission': 1.00, 'slippage': 0.5},
        'Current V6 Base': {'commission': 2.50, 'slippage': 0.75},
        'Higher Cost Broker': {'commission': 4.00, 'slippage': 1.0},
        'Market Stress': {'commission': 2.50, 'slippage': 1.5},    # Higher slippage
        'Extreme Stress': {'commission': 5.00, 'slippage': 2.0},   # Both high
        'Poor Execution': {'commission': 2.50, 'slippage': 2.5},   # Very poor fills
    }
    
    print("💰 V6 BAYESIAN TRANSACTION COST STRESS TEST")
    print("=" * 50)
    
    results = []
    
    for scenario_name, costs in cost_scenarios.items():
        print(f"\nTesting {scenario_name} scenario...")
        print(f"  Commission: ${costs['commission']:.2f}, Slippage: {costs['slippage']} ticks")
        
        # Calculate new transaction costs for each trade
        scenario_results = calculate_scenario_performance(df, costs['commission'], costs['slippage'], tick_value)
        
        results.append({
            'scenario': scenario_name,
            'commission': costs['commission'],
            'slippage_ticks': costs['slippage'],
            'mean_return': scenario_results['mean_return'],
            'mean_position_return': scenario_results['mean_position_return'],
            'win_rate': scenario_results['win_rate'],
            'profitable_trades': scenario_results['profitable_trades'],
            'total_trades': len(df),
            'profit_after_costs': scenario_results['profit_after_costs'],
            'cost_per_trade': scenario_results['avg_cost_per_trade'],
            'return_vs_base': scenario_results['return_vs_base']
        })
    
    results_df = pd.DataFrame(results)
    
    # Analysis and visualization
    create_cost_stress_visualizations(results_df)
    print_cost_stress_summary(results_df)
    
    # Save detailed results
    results_df.to_csv('./data/v6_cost_stress_results.csv', index=False)
    print(f"\n💾 Transaction cost stress results saved to: ./data/v6_cost_stress_results.csv")
    
    return results_df

def calculate_scenario_performance(df, commission, slippage_ticks, tick_value):
    """
    Calculate performance metrics under specific cost scenario
    """
    # Calculate new transaction costs
    def new_transaction_cost(position_multiplier):
        commission_cost = commission * position_multiplier
        slippage_cost = slippage_ticks * tick_value * position_multiplier
        return (commission_cost + slippage_cost) / (5000 * position_multiplier)
    
    # Recalculate returns with new costs
    df['new_transaction_cost'] = df['position_multiplier'].apply(new_transaction_cost)
    df['new_net_return'] = df['gross_return'] - df['new_transaction_cost']
    df['new_position_return'] = df['new_net_return'] * df['bayesian_multiplier']
    
    # Calculate metrics
    mean_return = df['new_net_return'].mean()
    mean_position_return = df['new_position_return'].mean()
    win_rate = (df['new_net_return'] > 0).mean()
    profitable_trades = (df['new_net_return'] > 0).sum()
    profit_after_costs = df['new_net_return'].sum()
    avg_cost_per_trade = df['new_transaction_cost'].mean()
    
    # Compare to base
    base_return = df['net_return'].mean()
    return_vs_base = (mean_return / base_return - 1) * 100 if base_return != 0 else 0
    
    return {
        'mean_return': mean_return,
        'mean_position_return': mean_position_return,
        'win_rate': win_rate,
        'profitable_trades': profitable_trades,
        'profit_after_costs': profit_after_costs,
        'avg_cost_per_trade': avg_cost_per_trade,
        'return_vs_base': return_vs_base
    }

def create_cost_stress_visualizations(results_df):
    """
    Create comprehensive visualizations of cost stress testing
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('V6 Bayesian Transaction Cost Stress Testing', fontsize=16, fontweight='bold')
    
    # 1. Returns vs Transaction Costs
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(results_df)), results_df['mean_position_return'] * 100, 
                    color=['green' if x > 0 else 'red' for x in results_df['mean_position_return']])
    ax1.set_xlabel('Cost Scenario')
    ax1.set_ylabel('Mean Position-Adjusted Return (%)')
    ax1.set_title('Performance Under Different Cost Scenarios')
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['scenario'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 2. Win Rate Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(results_df)), results_df['win_rate'] * 100, 
                    color='steelblue', alpha=0.7)
    ax2.set_xlabel('Cost Scenario')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate Stability Across Cost Scenarios')
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['scenario'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost per Trade Impact
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(results_df)), results_df['cost_per_trade'] * 100, 
                    color='orange', alpha=0.7)
    ax3.set_xlabel('Cost Scenario')
    ax3.set_ylabel('Average Cost per Trade (%)')
    ax3.set_title('Transaction Cost Impact')
    ax3.set_xticks(range(len(results_df)))
    ax3.set_xticklabels(results_df['scenario'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Return Degradation
    ax4 = axes[1, 1]
    colors = ['green' if x >= -10 else 'orange' if x >= -25 else 'red' for x in results_df['return_vs_base']]
    bars4 = ax4.bar(range(len(results_df)), results_df['return_vs_base'], color=colors, alpha=0.7)
    ax4.set_xlabel('Cost Scenario')
    ax4.set_ylabel('Return Change vs Base (%)')
    ax4.set_title('Performance Degradation')
    ax4.set_xticks(range(len(results_df)))
    ax4.set_xticklabels(results_df['scenario'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='Warning Threshold (-20%)')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the chart
    fig.savefig('./data/v6_transaction_cost_stress_test.png', dpi=300, bbox_inches='tight')
    print(f"\nCost stress test charts saved to: ./data/v6_transaction_cost_stress_test.png")
    
    return fig

def print_cost_stress_summary(results_df):
    """
    Print comprehensive cost stress testing summary
    """
    print(f"\n📊 TRANSACTION COST STRESS TEST SUMMARY")
    print("=" * 50)
    
    # Base scenario performance
    base_scenario = results_df[results_df['scenario'] == 'Current V6 Base'].iloc[0]
    print(f"Base V6 Performance:")
    print(f"  Mean Position-Adjusted Return: {base_scenario['mean_position_return']:.4%}")
    print(f"  Win Rate: {base_scenario['win_rate']:.1%}")
    print(f"  Cost per Trade: {base_scenario['cost_per_trade']:.4%}")
    
    # Find best and worst scenarios
    best_scenario = results_df.loc[results_df['mean_position_return'].idxmax()]
    worst_scenario = results_df.loc[results_df['mean_position_return'].idxmin()]
    
    print(f"\nBest Case Scenario: {best_scenario['scenario']}")
    print(f"  Return: {best_scenario['mean_position_return']:.4%} (+{best_scenario['return_vs_base']:+.1f}%)")
    print(f"  Commission: ${best_scenario['commission']:.2f}, Slippage: {best_scenario['slippage_ticks']} ticks")
    
    print(f"\nWorst Case Scenario: {worst_scenario['scenario']}")
    print(f"  Return: {worst_scenario['mean_position_return']:.4%} ({worst_scenario['return_vs_base']:+.1f}%)")
    print(f"  Commission: ${worst_scenario['commission']:.2f}, Slippage: {worst_scenario['slippage_ticks']} ticks")
    
    # Profitability analysis
    profitable_scenarios = results_df[results_df['mean_position_return'] > 0]
    print(f"\nProfitability Analysis:")
    print(f"  Profitable scenarios: {len(profitable_scenarios)}/{len(results_df)}")
    print(f"  Strategy remains profitable in {len(profitable_scenarios)/len(results_df)*100:.0f}% of cost scenarios")
    
    # Risk thresholds
    severe_degradation = results_df[results_df['return_vs_base'] < -25]
    moderate_degradation = results_df[(results_df['return_vs_base'] < -10) & (results_df['return_vs_base'] >= -25)]
    
    print(f"\nRisk Assessment:")
    if len(severe_degradation) == 0:
        print(f"  ✅ No scenarios show severe degradation (>25% loss)")
    else:
        print(f"  ⚠️  {len(severe_degradation)} scenarios show severe degradation")
        
    if len(moderate_degradation) == 0:
        print(f"  ✅ No scenarios show moderate degradation (10-25% loss)")
    else:
        print(f"  ⚠️  {len(moderate_degradation)} scenarios show moderate degradation")
    
    # Cost sensitivity
    cost_sensitivity = results_df['return_vs_base'].std()
    print(f"\nCost Sensitivity: {cost_sensitivity:.1f}% standard deviation")
    if cost_sensitivity < 15:
        print(f"  ✅ Strategy shows LOW sensitivity to transaction costs")
    elif cost_sensitivity < 25:
        print(f"  ⚠️  Strategy shows MODERATE sensitivity to transaction costs")
    else:
        print(f"  ❌ Strategy shows HIGH sensitivity to transaction costs")

def test_slippage_models(df):
    """
    Test different slippage models (fixed vs volume-dependent)
    """
    print(f"\n🎯 ADVANCED SLIPPAGE MODELING")
    print("=" * 35)
    
    # Volume-dependent slippage model
    def volume_dependent_slippage(volume_ratio, base_slippage=0.75):
        """Higher volume clusters may have more slippage"""
        if volume_ratio >= 50:
            return base_slippage * 1.5  # 50% more slippage for extreme volume
        elif volume_ratio >= 20:
            return base_slippage * 1.2  # 20% more slippage for high volume
        else:
            return base_slippage
    
    # Time-dependent slippage model
    def time_dependent_slippage(hour, base_slippage=0.75):
        """Market open/close times may have more slippage"""
        if hour in [9, 10, 15, 16]:  # Open/close hours
            return base_slippage * 1.3
        elif hour in [11, 12, 13, 14]:  # Mid-day quiet
            return base_slippage * 0.8
        else:
            return base_slippage
    
    # Test both models
    models = {
        'Fixed Slippage (Base)': lambda row: 0.75,
        'Volume-Dependent': lambda row: volume_dependent_slippage(row['volume_ratio']),
        'Time-Dependent': lambda row: time_dependent_slippage(pd.to_datetime(row['entry_time']).hour),
        'Combined Model': lambda row: max(
            volume_dependent_slippage(row['volume_ratio']),
            time_dependent_slippage(pd.to_datetime(row['entry_time']).hour)
        )
    }
    
    model_results = []
    
    for model_name, slippage_func in models.items():
        # Calculate slippage for each trade
        df['model_slippage'] = df.apply(slippage_func, axis=1)
        
        # Recalculate costs
        def new_cost(row):
            commission = 2.50 * row['position_multiplier']
            slippage = row['model_slippage'] * 12.50 * row['position_multiplier']
            return (commission + slippage) / (5000 * row['position_multiplier'])
        
        df['model_cost'] = df.apply(new_cost, axis=1)
        df['model_return'] = df['gross_return'] - df['model_cost']
        df['model_position_return'] = df['model_return'] * df['bayesian_multiplier']
        
        model_results.append({
            'model': model_name,
            'mean_return': df['model_position_return'].mean(),
            'avg_slippage': df['model_slippage'].mean(),
            'max_slippage': df['model_slippage'].max(),
            'cost_per_trade': df['model_cost'].mean()
        })
    
    model_df = pd.DataFrame(model_results)
    
    print("Slippage Model Comparison:")
    for _, row in model_df.iterrows():
        print(f"  {row['model']}: {row['mean_return']:.4%} return")
        print(f"    Avg slippage: {row['avg_slippage']:.2f} ticks, Cost: {row['cost_per_trade']:.4%}")
    
    return model_df

def main():
    """
    Run comprehensive transaction cost stress testing
    """
    print("💰 V6 BAYESIAN TRANSACTION COST ROBUSTNESS SUITE")
    print("=" * 60)
    
    # 1. Main stress test
    results_df = stress_test_transaction_costs()
    
    # 2. Advanced slippage modeling
    df = pd.read_csv("./data/backtest_results_v6.csv")
    slippage_results = test_slippage_models(df)
    
    print(f"\n🎯 TRANSACTION COST ROBUSTNESS COMPLETE")
    print(f"   Strategy cost sensitivity: {'HIGH' if results_df['mean_position_return'].std() > 0.003 else 'LOW'}")
    print(f"   Recommended for production: {'⚠️  WITH CAUTION' if (results_df['mean_position_return'] < 0).any() else '✅ YES'}")
    
    return results_df

if __name__ == "__main__":
    main() 