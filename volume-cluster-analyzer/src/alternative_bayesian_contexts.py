"""
V6 Bayesian Alternative Context Testing
Tests different contextual frameworks for Bayesian adaptive position sizing
to potentially enhance the already extraordinary 99% performance improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_alternative_bayesian_contexts(results_file="./data/backtest_results_v6.csv"):
    """
    Test various alternative Bayesian contexts beyond modal_bin
    """
    
    df = pd.read_csv(results_file)
    
    # Current V6 performance baseline
    baseline_return = df['position_adjusted_return'].mean()
    
    print("🧠 V6 BAYESIAN ALTERNATIVE CONTEXT TESTING")
    print("=" * 55)
    print(f"Current V6 modal_bin context: {baseline_return:.4%} per trade")
    print("Testing alternative contextual frameworks...")
    print()
    
    # Define alternative contexts
    contexts = {
        'modal_bin': create_modal_bin_context,
        'volume_rank': create_volume_rank_context,
        'time_of_day': create_time_of_day_context,
        'market_regime': create_market_regime_context,
        'signal_strength': create_signal_strength_context,
        'combined_context': create_combined_context,
        'momentum_context': create_momentum_context,
        'volatility_context': create_volatility_context
    }
    
    context_results = []
    
    for context_name, context_func in contexts.items():
        print(f"Testing {context_name} context...")
        
        # Create context values
        df_test = df.copy()
        df_test['test_context'] = context_func(df_test)
        
        # Simulate Bayesian performance with this context
        performance = simulate_bayesian_context(df_test, context_name)
        
        context_results.append({
            'context': context_name,
            'mean_return': performance['mean_return'],
            'return_vs_baseline': performance['return_vs_baseline'],
            'unique_contexts': performance['unique_contexts'],
            'avg_multiplier': performance['avg_multiplier'],
            'max_multiplier': performance['max_multiplier'],
            'bayesian_utilization': performance['bayesian_utilization'],
            'context_distribution': performance['context_distribution']
        })
    
    results_df = pd.DataFrame(context_results)
    
    # Analysis and visualization
    create_context_comparison_visualizations(results_df, baseline_return)
    print_context_analysis_summary(results_df, baseline_return)
    
    return results_df

def create_modal_bin_context(df):
    """Original modal_bin context (baseline)"""
    return df['modal_position'].apply(lambda x: min(int(x * 10), 9))

def create_volume_rank_context(df):
    """Volume rank based context"""
    return df['volume_rank'].fillna(1).astype(int)

def create_time_of_day_context(df):
    """Time of day context (market session)"""
    def get_time_context(entry_time):
        try:
            hour = pd.to_datetime(entry_time).hour
            if 9 <= hour <= 10:
                return 0  # Market open
            elif 11 <= hour <= 13:
                return 1  # Mid-morning
            elif 14 <= hour <= 15:
                return 2  # Afternoon
            else:
                return 3  # Extended hours
        except:
            return 1  # Default
    
    return df['entry_time'].apply(get_time_context)

def create_market_regime_context(df):
    """Market regime based on recent volatility"""
    def get_regime_context(volume_ratio):
        if volume_ratio >= 50:
            return 3  # Extreme volatility
        elif volume_ratio >= 20:
            return 2  # High volatility
        elif volume_ratio >= 10:
            return 1  # Medium volatility
        else:
            return 0  # Low volatility
    
    return df['volume_ratio'].apply(get_regime_context)

def create_signal_strength_context(df):
    """Signal strength based context"""
    return pd.cut(df['signal_strength'], bins=4, labels=[0, 1, 2, 3]).astype(int)

def create_combined_context(df):
    """Combined modal position and volume regime"""
    modal_bin = df['modal_position'].apply(lambda x: min(int(x * 5), 4))  # 0-4
    volume_regime = pd.cut(df['volume_ratio'], bins=3, labels=[0, 1, 2]).astype(int)
    return modal_bin * 3 + volume_regime  # Creates 0-14 range

def create_momentum_context(df):
    """Momentum based context"""
    # Use momentum if available, otherwise create from modal position
    if 'momentum' in df.columns:
        return pd.cut(df['momentum'], bins=4, labels=[0, 1, 2, 3]).astype(int)
    else:
        # Proxy momentum from modal position (extreme positions = strong momentum)
        return df['modal_position'].apply(lambda x: 0 if x <= 0.25 else 1 if x <= 0.75 else 2)

def create_volatility_context(df):
    """Create context based on market volatility"""
    df_copy = df.copy()
    # Calculate rolling volatility
    df_copy['vol_ma'] = df_copy['volume_ratio'].rolling(window=10, min_periods=1).std()
    # Handle NaN values before converting to int
    vol_context = pd.cut(df_copy['vol_ma'], bins=3, labels=[0, 1, 2])
    return vol_context.fillna(1).astype(int)  # Fill NaN with middle value

def simulate_bayesian_context(df, context_name):
    """
    Simulate Bayesian performance with alternative context
    """
    # V6 parameters
    SCALING_FACTOR = 6.0
    BAYESIAN_MAX_MULTIPLIER = 3.0
    MIN_TRADES_FOR_BAYESIAN = 3
    ALPHA_PRIOR = 1.0
    BETA_PRIOR = 1.0
    
    # Build context statistics (simulating historical accumulation)
    context_stats = {}
    simulated_returns = []
    simulated_multipliers = []
    utilization_count = 0
    
    for idx, row in df.iterrows():
        context_value = row['test_context']
        
        # Get historical stats for this context (only past trades)
        if context_value not in context_stats:
            context_stats[context_value] = {'wins': 0, 'losses': 0, 'total': 0}
        
        # Calculate Bayesian multiplier based on current stats
        stats = context_stats[context_value]
        
        if stats['total'] < MIN_TRADES_FOR_BAYESIAN:
            # Insufficient data - use conservative sizing
            bayesian_multiplier = 1.0
        else:
            # Calculate posterior parameters
            alpha_post = ALPHA_PRIOR + stats['wins']
            beta_post = BETA_PRIOR + stats['losses']
            
            # Expected win probability
            expected_p = alpha_post / (alpha_post + beta_post)
            
            # Position multiplier
            if expected_p > 0.5:
                raw_multiplier = 1.0 + (expected_p - 0.5) * SCALING_FACTOR
                bayesian_multiplier = min(raw_multiplier, BAYESIAN_MAX_MULTIPLIER)
                utilization_count += 1
            else:
                bayesian_multiplier = 1.0
        
        # Calculate position-adjusted return
        position_return = row['net_return'] * bayesian_multiplier
        simulated_returns.append(position_return)
        simulated_multipliers.append(bayesian_multiplier)
        
        # Update context statistics with this trade result (for future trades)
        if row['net_return'] > 0:
            context_stats[context_value]['wins'] += 1
        else:
            context_stats[context_value]['losses'] += 1
        context_stats[context_value]['total'] += 1
    
    # Calculate performance metrics
    mean_return = np.mean(simulated_returns)
    baseline_return = df['position_adjusted_return'].mean()
    return_vs_baseline = (mean_return / baseline_return - 1) * 100 if baseline_return != 0 else 0
    
    # Context distribution analysis
    context_distribution = df['test_context'].value_counts().to_dict()
    
    return {
        'mean_return': mean_return,
        'return_vs_baseline': return_vs_baseline,
        'unique_contexts': len(context_stats),
        'avg_multiplier': np.mean(simulated_multipliers),
        'max_multiplier': np.max(simulated_multipliers),
        'bayesian_utilization': utilization_count / len(df),
        'context_distribution': context_distribution
    }

def create_context_comparison_visualizations(results_df, baseline_return):
    """
    Create comprehensive visualizations comparing different contexts
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V6 Bayesian Alternative Context Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Return comparison
    ax1 = axes[0, 0]
    colors = ['red' if x == 'modal_bin' else 'green' if y > 0 else 'orange' 
              for x, y in zip(results_df['context'], results_df['return_vs_baseline'])]
    bars1 = ax1.bar(range(len(results_df)), results_df['mean_return'] * 100, color=colors, alpha=0.7)
    ax1.axhline(y=baseline_return * 100, color='red', linestyle='--', alpha=0.8, 
                label=f'Current V6 ({baseline_return:.3%})')
    ax1.set_xlabel('Context Type')
    ax1.set_ylabel('Mean Position-Adjusted Return (%)')
    ax1.set_title('Return Performance by Context')
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['context'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Return improvement vs baseline
    ax2 = axes[0, 1]
    colors2 = ['red' if x == 'modal_bin' else 'green' if y > 5 else 'orange' if y > 0 else 'red'
               for x, y in zip(results_df['context'], results_df['return_vs_baseline'])]
    bars2 = ax2.bar(range(len(results_df)), results_df['return_vs_baseline'], color=colors2, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Significant Improvement (+5%)')
    ax2.set_xlabel('Context Type')
    ax2.set_ylabel('Return Change vs Current V6 (%)')
    ax2.set_title('Performance Improvement vs Current V6')
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['context'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Context diversity vs performance
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['unique_contexts'], results_df['return_vs_baseline'], 
                         s=100, alpha=0.7, c=results_df['bayesian_utilization'], 
                         cmap='viridis', edgecolors='black')
    ax3.set_xlabel('Number of Unique Contexts')
    ax3.set_ylabel('Return Change vs V6 (%)')
    ax3.set_title('Context Diversity vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # Add context labels
    for i, row in results_df.iterrows():
        ax3.annotate(row['context'][:8], (row['unique_contexts'], row['return_vs_baseline']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Bayesian Utilization Rate')
    
    # 4. Position sizing effectiveness
    ax4 = axes[1, 1]
    ax4.scatter(results_df['avg_multiplier'], results_df['return_vs_baseline'], 
               s=results_df['bayesian_utilization']*200, alpha=0.7, 
               c=results_df['unique_contexts'], cmap='plasma', edgecolors='black')
    ax4.set_xlabel('Average Bayesian Multiplier')
    ax4.set_ylabel('Return Change vs V6 (%)')
    ax4.set_title('Position Sizing vs Performance\n(Size = Utilization Rate)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    fig.savefig('./data/v6_alternative_contexts_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nAlternative contexts charts saved to: ./data/v6_alternative_contexts_analysis.png")
    
    return fig

def print_context_analysis_summary(results_df, baseline_return):
    """
    Print comprehensive context analysis summary
    """
    print(f"\n🎯 ALTERNATIVE CONTEXT ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Current V6 baseline
    current_v6 = results_df[results_df['context'] == 'modal_bin'].iloc[0]
    print(f"Current V6 modal_bin baseline:")
    print(f"  Return: {current_v6['mean_return']:.4%}")
    print(f"  Unique contexts: {current_v6['unique_contexts']}")
    print(f"  Bayesian utilization: {current_v6['bayesian_utilization']:.1%}")
    
    # Find best alternatives
    alternatives = results_df[results_df['context'] != 'modal_bin']
    best_alternative = alternatives.loc[alternatives['mean_return'].idxmax()]
    
    print(f"\nBest alternative context: {best_alternative['context']}")
    print(f"  Return: {best_alternative['mean_return']:.4%} ({best_alternative['return_vs_baseline']:+.1f}%)")
    print(f"  Unique contexts: {best_alternative['unique_contexts']}")
    print(f"  Bayesian utilization: {best_alternative['bayesian_utilization']:.1%}")
    print(f"  Avg multiplier: {best_alternative['avg_multiplier']:.3f}")
    
    # Improvement analysis
    significant_improvements = alternatives[alternatives['return_vs_baseline'] > 5]
    moderate_improvements = alternatives[(alternatives['return_vs_baseline'] > 0) & 
                                       (alternatives['return_vs_baseline'] <= 5)]
    
    print(f"\nImprovement Analysis:")
    print(f"  Significant improvements (>5%): {len(significant_improvements)}")
    print(f"  Moderate improvements (0-5%): {len(moderate_improvements)}")
    print(f"  No improvement: {len(alternatives) - len(significant_improvements) - len(moderate_improvements)}")
    
    if len(significant_improvements) > 0:
        print(f"\n🚀 CONTEXTS WITH SIGNIFICANT IMPROVEMENT:")
        for _, row in significant_improvements.iterrows():
            print(f"  {row['context']}: {row['return_vs_baseline']:+.1f}% improvement")
    
    # Context diversity analysis
    print(f"\nContext Diversity Analysis:")
    print(f"  Most diverse: {results_df.loc[results_df['unique_contexts'].idxmax(), 'context']} " +
          f"({results_df['unique_contexts'].max()} contexts)")
    print(f"  Least diverse: {results_df.loc[results_df['unique_contexts'].idxmin(), 'context']} " +
          f"({results_df['unique_contexts'].min()} contexts)")
    
    # Utilization efficiency
    high_util = results_df[results_df['bayesian_utilization'] > 0.9]
    print(f"\nHigh Bayesian Utilization (>90%): {len(high_util)} contexts")
    for _, row in high_util.iterrows():
        print(f"  {row['context']}: {row['bayesian_utilization']:.1%} utilization, " +
              f"{row['return_vs_baseline']:+.1f}% vs baseline")

def test_hybrid_contexts(df):
    """
    Test hybrid combinations of the best performing contexts
    """
    print(f"\n🔬 HYBRID CONTEXT TESTING")
    print("=" * 30)
    
    # Test combinations of promising contexts
    hybrid_contexts = {
        'modal_volume': lambda df: df['modal_position'].apply(lambda x: min(int(x * 5), 4)) * 4 + 
                                  pd.cut(df['volume_ratio'], bins=4, labels=[0,1,2,3]).astype(int),
        
        'time_signal': lambda df: create_time_of_day_context(df) * 4 + 
                                 pd.cut(df['signal_strength'], bins=4, labels=[0,1,2,3]).astype(int),
        
        'regime_modal': lambda df: create_market_regime_context(df) * 5 + 
                                  df['modal_position'].apply(lambda x: min(int(x * 5), 4)),
        
        'triple_context': lambda df: (df['modal_position'].apply(lambda x: min(int(x * 3), 2)) * 9 + 
                                     create_market_regime_context(df) * 3 + 
                                     create_time_of_day_context(df))
    }
    
    hybrid_results = []
    baseline_return = df['position_adjusted_return'].mean()
    
    for context_name, context_func in hybrid_contexts.items():
        df_test = df.copy()
        df_test['test_context'] = context_func(df_test)
        
        performance = simulate_bayesian_context(df_test, context_name)
        
        hybrid_results.append({
            'context': context_name,
            'mean_return': performance['mean_return'],
            'return_vs_baseline': performance['return_vs_baseline'],
            'unique_contexts': performance['unique_contexts'],
            'bayesian_utilization': performance['bayesian_utilization']
        })
    
    hybrid_df = pd.DataFrame(hybrid_results)
    
    print("Hybrid Context Results:")
    for _, row in hybrid_df.iterrows():
        print(f"  {row['context']}: {row['mean_return']:.4%} ({row['return_vs_baseline']:+.1f}%)")
        print(f"    {row['unique_contexts']} contexts, {row['bayesian_utilization']:.1%} utilization")
    
    return hybrid_df

def main():
    """
    Run comprehensive alternative context testing
    """
    print("🧠 V6 BAYESIAN ALTERNATIVE CONTEXT EXPLORATION")
    print("=" * 60)
    
    # 1. Test all alternative contexts
    context_results = test_alternative_bayesian_contexts()
    
    # 2. Test hybrid contexts
    df = pd.read_csv("./data/backtest_results_v6.csv")
    hybrid_results = test_hybrid_contexts(df)
    
    # 3. Save results
    context_results.to_csv('./data/v6_alternative_contexts_results.csv', index=False)
    hybrid_results.to_csv('./data/v6_hybrid_contexts.csv', index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   Alternative contexts: ./data/v6_alternative_contexts_results.csv")
    print(f"   Hybrid contexts: ./data/v6_hybrid_contexts.csv")
    
    # 4. Final recommendation
    best_alternative = context_results[context_results['context'] != 'modal_bin'].loc[
        context_results[context_results['context'] != 'modal_bin']['mean_return'].idxmax()
    ]
    
    best_hybrid = hybrid_results.loc[hybrid_results['mean_return'].idxmax()]
    
    print(f"\n🎯 CONTEXT OPTIMIZATION RECOMMENDATIONS:")
    print(f"   Current V6 modal_bin: Excellent baseline performance")
    print(f"   Best alternative: {best_alternative['context']} ({best_alternative['return_vs_baseline']:+.1f}%)")
    print(f"   Best hybrid: {best_hybrid['context']} ({best_hybrid['return_vs_baseline']:+.1f}%)")
    
    if best_alternative['return_vs_baseline'] > 10 or best_hybrid['return_vs_baseline'] > 10:
        print(f"   ✅ SIGNIFICANT OPPORTUNITY: Consider implementing best alternative")
    elif best_alternative['return_vs_baseline'] > 5 or best_hybrid['return_vs_baseline'] > 5:
        print(f"   💡 MODERATE OPPORTUNITY: Consider testing in paper trading")
    else:
        print(f"   ✅ CURRENT OPTIMAL: modal_bin context is near-optimal for this dataset")

if __name__ == "__main__":
    main() 