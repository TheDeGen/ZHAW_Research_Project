# 3.7 Backtesting Framework

This section details the backtesting methodology employed to translate model predictions into simulated trading outcomes and evaluate economic performance. While Section 3.6 assessed predictive accuracy through classification metrics, this section evaluates the practical utility of the signal-generating pipeline by simulating trading execution on held-out test data and comparing performance against benchmark strategies.

---

## 3.7.1 Trading Strategy Specification

**Purpose:** Define how model predictions are translated into executable trading actions and establish the benchmark strategies for comparison.

**Key points to cover:**

### Signal Model Strategy

- **Action mapping**: Model predictions (Long/Neutral/Short) are translated directly into trading positions:
  - **Long (+1)**: Buy at day-ahead auction, sell at intraday spot — profits when spot > day-ahead
  - **Short (−1)**: Sell at day-ahead auction, buy at intraday spot — profits when spot < day-ahead
  - **Neutral (0)**: No position taken — neither profit nor loss
  
- **Signal generation process** (two-stage pipeline):
  1. XGBoost base model generates calibrated probability estimates for spread direction
  2. LightGBM meta-learner refines these into final Long/Neutral/Short signals
  
- Note that the Neutral class serves a risk management function, allowing the model to abstain when prediction confidence is insufficient

### Baseline Model Strategy

- Uses the price-only LightGBM classifier (without news-derived features) as the baseline
- Same action mapping as the signal model
- Enables direct comparison of news feature contribution to trading performance

### Naïve Benchmark Strategy

- **Always-long strategy**: Systematically buys at day-ahead auction and sells at intraday spot for every observation
- Equivalent to setting action = +1 for all timestamps
- Represents a simple market-making approach that profits from average positive spreads but incurs losses during negative spread periods
- Provides a lower bound for model utility—if the signal model cannot outperform this naïve strategy, the predictive complexity adds no economic value

### Transaction Cost Modelling

- **Fixed costs**: Per-trade fee in EUR/MWh representing exchange fees, clearing costs, and broker commissions
- **Proportional costs**: Percentage of trade value representing bid-ask spreads and market impact
- Returns computation:
  $$R_t = \begin{cases} 
  \text{spread}_t - c_{\text{fixed}} - c_{\%} \cdot |\text{spread}_t| & \text{if action} = +1 \\
  -\text{spread}_t - c_{\text{fixed}} - c_{\%} \cdot |\text{spread}_t| & \text{if action} = -1 \\
  0 & \text{if action} = 0
  \end{cases}$$
- Note that transaction costs are only incurred when positions are taken (Long or Short), not for Neutral signals

---

## 3.7.2 Performance Metrics

**Purpose:** Define the economic performance metrics used to evaluate trading strategy outcomes.

**Key points to cover:**

### Return-Based Metrics

- **Total Return**: Cumulative profit/loss over the test period in EUR/MWh
  $$\text{Total Return} = \sum_{t=1}^{T} R_t$$
  
- **Average Return**: Mean return per period, indicating expected profit per trade
  $$\bar{R} = \frac{1}{T} \sum_{t=1}^{T} R_t$$

- **Volatility**: Standard deviation of returns, measuring return dispersion
  $$\sigma = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (R_t - \bar{R})^2}$$

### Risk-Adjusted Metrics

- **Sharpe Ratio** (annualised): Return per unit of total risk, the standard risk-adjusted performance measure (Sharpe, 1966)
  $$\text{Sharpe} = \frac{\bar{R}}{\sigma} \cdot \sqrt{N_{\text{periods/year}}}$$
  where $N_{\text{periods/year}} = 24 \times 365 = 8{,}760$ for hourly data
  
- **Sortino Ratio** (annualised): Return per unit of downside risk, penalising only negative volatility (Sortino & van der Meer, 1991)
  $$\text{Sortino} = \frac{\bar{R}}{\sigma_{\text{downside}}} \cdot \sqrt{N_{\text{periods/year}}}$$
  where $\sigma_{\text{downside}}$ is computed only from negative returns
  - Rationale: Upside volatility is desirable in trading; Sortino isolates harmful downside risk

### Drawdown Analysis

- **Maximum Drawdown**: Largest peak-to-trough decline in cumulative returns
  $$\text{MDD} = \min_t \left( \text{Cumulative}_t - \max_{s \leq t} \text{Cumulative}_s \right)$$
  - Measures worst-case capital erosion during the strategy's lifetime
  - Critical for assessing tail risk and capital requirements

### Trade Statistics

- **Win Rate**: Percentage of profitable trades among all executed trades
  $$\text{Win Rate} = \frac{\text{Number of } R_t > 0}{\text{Number of } R_t \neq 0} \times 100\%$$
  
- **Number of Trades**: Total count of non-zero positions (Long + Short signals)
  - Lower trade counts with similar returns indicate more efficient signal generation

---

## 3.7.3 Comparative Analysis Framework

**Purpose:** Establish the methodology for comparing strategies and interpreting results.

**Key points to cover:**

### Strategy Comparison Table

- Present side-by-side comparison of all three strategies:
  1. LightGBM Signal Model (with news features)
  2. LightGBM Baseline Model (price-only)
  3. Naïve Always-Long Benchmark

- Comparison dimensions:
  - Total Return
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Win Rate
  - Number of Trades

### Interpretation Guidelines

- **Signal vs. Baseline**: Isolates the marginal contribution of news-derived features
  - If signal model outperforms baseline, news features add predictive value beyond price history
  - Consistent with ablation analysis in Section 3.5.5
  
- **Signal vs. Naïve**: Establishes whether model complexity provides economic benefit
  - Naïve strategy captures average market conditions; model should exploit deviations
  - Significant outperformance validates the entire pipeline from data collection through signal generation

- **Risk-return trade-offs**: A model may achieve higher returns but with proportionally higher risk
  - Sharpe and Sortino ratios normalise for risk, enabling fair comparison
  - Maximum drawdown indicates worst-case scenarios that may be unacceptable regardless of average returns

### Equity Curve Visualisation

- Plot cumulative returns over time for all strategies
- Enables visual identification of:
  - Periods of sustained outperformance/underperformance
  - Regime changes where model effectiveness varies
  - Drawdown events and recovery patterns

### Sensitivity Analysis (Optional Extension)

- Evaluate performance under varying transaction cost assumptions
- Test robustness of strategy ranking to cost parameter changes
- Identify break-even transaction costs where model loses edge over naïve benchmark

---

## Key Implementation Notes (for Results section)

The following elements from `evaluation.py` should inform the Results presentation:

1. **`setup_backtest_strategies()`**: Initialises the three strategies and computes spread series from test set price columns
2. **`actions_to_returns()`**: Converts action arrays to return series with configurable transaction costs
3. **`summarise_returns()`**: Computes all performance metrics for a single strategy
4. **`summarise_strategy_set()`**: Generates comparative DataFrame across all strategies
5. **`compute_strategy_returns()`**: Batch processes multiple strategies efficiently

---

## Transition to Results (Section 4)

Section 3.7 completes the methodological framework. The Results section will present:

1. **Classification performance** (from Section 3.6): Accuracy, F1, AUC with confidence intervals and statistical significance tests
2. **Backtesting results** (from Section 3.7): Strategy performance comparison table, equity curves, and risk metrics
3. **Feature importance analysis**: Which news topics and time-decay parameters contributed most to predictive performance

The combined evaluation—statistical predictive accuracy and economic trading performance—provides a comprehensive assessment of whether news-driven signals offer actionable value in German electricity markets.

---

## References (to be integrated into main bibliography)

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094

Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28. https://doi.org/10.3905/jpm.2015.42.1.013

Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138. https://doi.org/10.1086/294846

Sortino, F. A., & van der Meer, R. (1991). Downside risk. *Journal of Portfolio Management*, 17(4), 27–31. https://doi.org/10.3905/jpm.1991.409343

