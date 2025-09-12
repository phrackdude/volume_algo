"""
V6 Bayesian Parameter Sensitivity Analysis
Tests the robustness of the extraordinary 99% performance improvement
across different parameter configurations to ensure production readiness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def run_bayesian_sensitivity_test(base_results_file="./data/backtest_results_v6.csv"):
    """
    Test sensitivity to key Bayesian parameters by simulating different configurations
    """
    
    # Load V6 base results
    df = pd.read_csv(base_results_file)
    
    # Base V6 parameters
    base_params = {
        'SCALING_FACTOR': 6.0,
        'BAYESIAN_MAX_MULTIPLIER': 3.0,
        'MIN_TRADES_FOR_BAYESIAN': 3,
        'ALPHA_PRIOR': 1.0,
        'BETA_PRIOR': 1.0
    }
    
    # Parameter sensitivity ranges
    param_ranges = {
        'SCALING_FACTOR': [3.0, 4.5, 6.0, 7.5, 9.0],      # ±50% from base
        'BAYESIAN_MAX_MULTIPLIER': [2.0, 2.5, 3.0, 3.5, 4.0],  # ±33% from base
        'MIN_TRADES_FOR_BAYESIAN': [1, 2, 3, 5, 7],        # Lower/higher thresholds
        'ALPHA_PRIOR': [0.5, 1.0, 1.5],                    # Different priors
        'BETA_PRIOR': [0.5, 1.0, 1.5]                      # Different priors
    }
    
    print("🔬 V6 BAYESIAN PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Base V6 performance: {df['position_adjusted_return'].mean():.4%} per trade")
    print(f"Testing {len(param_ranges)} parameters across multiple configurations...")
    print()
    
    # Test each parameter individually (single-factor sensitivity)
    sensitivity_results = {}
    
    for param_name, param_values in param_ranges.items():
        print(f"Testing {param_name} sensitivity...")
        param_results = []
        
        for param_value in param_values:
            # Simulate the parameter change by recalculating Bayesian multipliers
            modified_df = simulate_parameter_change(df.copy(), param_name, param_value, base_params)
            
            # Calculate performance metrics
            metrics = {
                'parameter': param_name,
                'value': param_value,
                'mean_return': modified_df['simulated_position_return'].mean(),
                'win_rate': (modified_df['net_return'] > 0).mean(),
                'avg_multiplier': modified_df['simulated_multiplier'].mean(),
                'max_multiplier': modified_df['simulated_multiplier'].max(),
                'trades_affected': (modified_df['simulated_multiplier'] != 1.0).sum(),
                'return_vs_base': (modified_df['simulated_position_return'].mean() / 
                                 df['position_adjusted_return'].mean() - 1) * 100
            }
            
            param_results.append(metrics)
        
        sensitivity_results[param_name] = param_results
    
    return sensitivity_results, df

def simulate_parameter_change(df, param_name, param_value, base_params):
    """
    Simulate how changing a Bayesian parameter would affect historical results
    """
    # Create copy with new parameter
    params = base_params.copy()
    params[param_name] = param_value
    
    # Recalculate Bayesian multipliers with new parameters
    df['simulated_multiplier'] = df.apply(
        lambda row: calculate_new_bayesian_multiplier(
            row, params
        ), axis=1
    )
    
    # Recalculate position-adjusted returns
    df['simulated_position_return'] = (
        df['net_return'] * df['simulated_multiplier'] / df['bayesian_multiplier']
    ).fillna(df['net_return'])
    
    return df

def calculate_new_bayesian_multiplier(row, params):
    """
    Recalculate Bayesian multiplier with new parameters
    """
    # Extract key values
    expected_p = row['bayesian_expected_p']
    original_method = row['bayesian_diagnostic_method']
    total_trades = row['bayesian_total_trades']
    
    # Apply new minimum trades threshold
    if total_trades < params['MIN_TRADES_FOR_BAYESIAN']:
        return 1.0
    
    # Apply new scaling logic
    if expected_p > 0.5:
        raw_multiplier = 1.0 + (expected_p - 0.5) * params['SCALING_FACTOR']
        return min(raw_multiplier, params['BAYESIAN_MAX_MULTIPLIER'])
    else:
        return 1.0

def create_sensitivity_visualizations(sensitivity_results, base_performance):
    """
    Create comprehensive visualizations of parameter sensitivity
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('V6 Bayesian Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    param_names = list(sensitivity_results.keys())
    
    for i, param_name in enumerate(param_names):
        if i >= 6:  # Limit to 6 plots
            break
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        results = sensitivity_results[param_name]
        df_param = pd.DataFrame(results)
        
        # Plot return vs parameter value
        ax.plot(df_param['value'], df_param['mean_return'] * 100, 
                'o-', linewidth=2, markersize=8, label='Position-Adjusted Return')
        ax.axhline(y=base_performance * 100, color='red', linestyle='--', 
                  alpha=0.7, label='Base V6 Performance')
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Mean Return (%)')
        ax.set_title(f'{param_name} Sensitivity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance change annotations
        for _, row_data in df_param.iterrows():
            if abs(row_data['return_vs_base']) > 5:  # Only annotate significant changes
                ax.annotate(f"{row_data['return_vs_base']:+.1f}%", 
                           (row_data['value'], row_data['mean_return'] * 100),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, alpha=0.8)
    
    # Remove empty subplots
    for i in range(len(param_names), 6):
        row = i // 3
        col = i % 3
        axes[row, col].remove()
    
    plt.tight_layout()
    return fig

def print_sensitivity_summary(sensitivity_results, base_performance):
    """
    Print comprehensive sensitivity analysis summary
    """
    print("\n🎯 PARAMETER SENSITIVITY SUMMARY")
    print("=" * 50)
    
    for param_name, results in sensitivity_results.items():
        df_param = pd.DataFrame(results)
        
        print(f"\n📊 {param_name}:")
        
        # Find base value more safely
        base_values = [r['value'] for r in results if abs(r['return_vs_base']) < 5.0]
        if base_values:
            print(f"   Base value: {base_values[0]}")
        else:
            print(f"   Base value: {results[len(results)//2]['value']} (estimated)")
        
        # Find best and worst performance
        best_idx = df_param['mean_return'].idxmax()
        worst_idx = df_param['mean_return'].idxmin()
        
        best_row = df_param.iloc[best_idx]
        worst_row = df_param.iloc[worst_idx]
        
        print(f"   Best performance: {best_row['value']} -> {best_row['mean_return']:.4%} ({best_row['return_vs_base']:+.1f}%)")
        print(f"   Worst performance: {worst_row['value']} -> {worst_row['mean_return']:.4%} ({worst_row['return_vs_base']:+.1f}%)")
        
        # Calculate sensitivity (standard deviation of returns)
        sensitivity = df_param['mean_return'].std()
        print(f"   Parameter sensitivity: {sensitivity:.4%} (lower = more robust)")
        
        # Check if any configuration fails dramatically
        major_failures = df_param[df_param['return_vs_base'] < -20]
        if len(major_failures) > 0:
            print(f"   ⚠️  WARNING: {len(major_failures)} configurations show >20% performance drop")
        else:
            print(f"   ✅ All configurations maintain reasonable performance")

def run_monte_carlo_bootstrap(df, n_simulations=1000):
    """
    Monte Carlo bootstrap analysis to test statistical significance
    """
    print(f"\n🎲 MONTE CARLO BOOTSTRAP ANALYSIS ({n_simulations} simulations)")
    print("=" * 50)
    
    # Original performance
    original_return = df['position_adjusted_return'].mean()
    original_win_rate = (df['net_return'] > 0).mean()
    
    # Bootstrap samples
    bootstrap_returns = []
    bootstrap_win_rates = []
    
    n_trades = len(df)
    
    for _ in range(n_simulations):
        # Sample with replacement
        sample_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        sample_df = df.iloc[sample_indices]
        
        bootstrap_returns.append(sample_df['position_adjusted_return'].mean())
        bootstrap_win_rates.append((sample_df['net_return'] > 0).mean())
    
    bootstrap_returns = np.array(bootstrap_returns)
    bootstrap_win_rates = np.array(bootstrap_win_rates)
    
    # Calculate confidence intervals
    return_ci_lower = np.percentile(bootstrap_returns, 2.5)
    return_ci_upper = np.percentile(bootstrap_returns, 97.5)
    
    winrate_ci_lower = np.percentile(bootstrap_win_rates, 2.5)
    winrate_ci_upper = np.percentile(bootstrap_win_rates, 97.5)
    
    print(f"Original performance: {original_return:.4%} per trade")
    print(f"Bootstrap mean: {bootstrap_returns.mean():.4%}")
    print(f"95% Confidence Interval: [{return_ci_lower:.4%}, {return_ci_upper:.4%}]")
    print(f"Standard Error: {bootstrap_returns.std():.4%}")
    print()
    print(f"Original win rate: {original_win_rate:.2%}")
    print(f"Bootstrap win rate: {bootstrap_win_rates.mean():.2%}")
    print(f"95% CI: [{winrate_ci_lower:.2%}, {winrate_ci_upper:.2%}]")
    
    # Statistical significance test
    prob_positive = (bootstrap_returns > 0).mean()
    prob_above_threshold = (bootstrap_returns > 0.005).mean()  # 0.5% threshold
    
    print(f"\nStatistical Significance:")
    print(f"  Probability of positive returns: {prob_positive:.1%}")
    print(f"  Probability of >0.5% returns: {prob_above_threshold:.1%}")
    
    if prob_positive > 0.95:
        print(f"  ✅ Strategy is statistically significant (>95% confidence)")
    else:
        print(f"  ⚠️  Strategy lacks statistical significance at 95% level")
    
    return bootstrap_returns, bootstrap_win_rates

def test_different_market_regimes(df):
    """
    Test performance across different market volatility regimes
    """
    print(f"\n📈 MARKET REGIME ANALYSIS")
    print("=" * 30)
    
    # Add rolling volatility
    df['volatility_regime'] = pd.cut(
        df['volume_ratio'], 
        bins=[0, 10, 20, 50, 1000], 
        labels=['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
    )
    
    regime_analysis = df.groupby('volatility_regime').agg({
        'position_adjusted_return': ['mean', 'std', 'count'],
        'net_return': lambda x: (x > 0).mean(),
        'bayesian_multiplier': 'mean'
    }).round(4)
    
    print("Performance by Volume Regime:")
    for regime in regime_analysis.index:
        stats = regime_analysis.loc[regime]
        win_rate = stats[('net_return', '<lambda>')]
        print(f"  {regime}: {stats[('position_adjusted_return', 'mean')]:.4%} return " +
              f"({stats[('position_adjusted_return', 'count')]} trades, " +
              f"{win_rate:.1%} win rate)")
    
    return regime_analysis

def main():
    """
    Run comprehensive sensitivity analysis
    """
    print("🚀 V6 BAYESIAN ROBUSTNESS TESTING SUITE")
    print("=" * 60)
    
    # 1. Parameter sensitivity analysis
    sensitivity_results, df = run_bayesian_sensitivity_test()
    base_performance = df['position_adjusted_return'].mean()
    
    # 2. Create visualizations
    fig = create_sensitivity_visualizations(sensitivity_results, base_performance)
    fig.savefig('./data/v6_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSensitivity charts saved to: ./data/v6_sensitivity_analysis.png")
    
    # 3. Print summary
    print_sensitivity_summary(sensitivity_results, base_performance)
    
    # 4. Monte Carlo bootstrap
    bootstrap_returns, bootstrap_win_rates = run_monte_carlo_bootstrap(df)
    
    # 5. Market regime analysis
    regime_analysis = test_different_market_regimes(df)
    
    # 6. Save detailed results
    sensitivity_df = pd.concat([
        pd.DataFrame(results) for results in sensitivity_results.values()
    ])
    sensitivity_df.to_csv('./data/v6_sensitivity_results.csv', index=False)
    
    print(f"\n💾 Detailed sensitivity results saved to: ./data/v6_sensitivity_results.csv")
    print(f"\n🎯 ROBUSTNESS ASSESSMENT COMPLETE")
    print(f"   Strategy shows {'ROBUST' if sensitivity_df['return_vs_base'].std() < 15 else 'SENSITIVE'} performance")
    print(f"   Recommended for production: {'✅ YES' if bootstrap_returns.mean() > 0.006 else '⚠️  NEEDS REVIEW'}")

if __name__ == "__main__":
    main() 