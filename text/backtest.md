## 3.7 Backtesting Framework

This section specifies the backtesting framework employed to evaluate the economic implications of the model-generated trading signals developed in Section 3.5. The implementation is fully deterministic and operates exclusively on the held-out test sample established in Section 3.4.6, translating discrete model outputs into executable positions and subsequently into profit-and-loss (P&L) under explicit assumptions regarding timing, position sizing, and transaction costs. The backtest is designed as a signal-to-portfolio pipeline, consistent with best practice in empirical asset pricing and systematic trading research, wherein the mapping from predictions to trades is fixed ex ante and performance is measured out of sample (Campbell, Lo, & MacKinlay, 1997, pp. 149–180; Chan, 2013, pp. 173–198; López de Prado, 2018, pp. 101–130).

### 3.7.1 Trading Strategy Specification

The LightGBM classifiers trained in Section 3.5.5 yield a discrete action $a_t \in \{-1, 0, +1\}$ at each hourly decision point $t$, corresponding to the three-class target variable defined in Section 3.4.1: $+1$ denotes a Long signal, $-1$ a Short signal, and $0$ a Neutral (no-trade) decision. These actions are translated into trades against the realised spread series—computed as the difference between spot and day-ahead prices—which serves as the basis for P&L calculation. Where a horizon-aligned spread series is provided in the dataset, this series is used directly; otherwise, the spread is computed from contemporaneous prices. Periods with undefined spread values (e.g., due to shifting at the end of the sample) are treated as non-trading periods, and the portfolio is carried forward unchanged. The mapping from predictions to actions is fixed prior to evaluation and applied uniformly across candidate strategies to preserve out-of-sample integrity (Campbell, Lo, & MacKinlay, 1997, pp. 149–180; Chan, 2013, pp. 173–198).

### 3.7.2 Portfolio Simulation and Position Sizing

Economic performance is evaluated using a self-financing portfolio simulation that commences from an initial capital base of €100,000 and rebalances at the hourly frequency. At each period $t$, if $a_t \neq 0$, the strategy allocates a fixed fraction $w$ of current portfolio value $V_t$ to the position (default: $w = 10\%$). The notional position size in euros is therefore:

$$
N_t = w \cdot V_t
$$

As electricity positions are naturally expressed in energy units, the framework converts the euro notional into traded volume (MWh) using the day-ahead price $D_t$ as a sizing reference. Specifically, the traded volume is:

$$
q_t = \frac{w \cdot V_t}{\max(|D_t|, \underline{D})}
$$

where $\underline{D}$ denotes a minimum price floor employed to prevent mechanically inflated volumes during episodes of very low (or near-zero) prices. In addition, a hard cap on volume per trade is imposed to reflect practical liquidity constraints and to avoid unrealistic leverage through unit scaling. Together, the price floor and volume cap operationalise conservative feasibility constraints commonly recommended in backtesting research to reduce sensitivity to extreme observations and microstructural artefacts (Chan, 2013, pp. 185–192; López de Prado, 2018, pp. 115–122).

### 3.7.3 Profit-and-Loss Calculation and Transaction Costs

Trade-level P&L is computed by applying the directional action to the realised spread and scaling by traded volume:

$$
\text{P\&L}^{\text{gross}}_t = a_t \cdot S_t \cdot q_t
$$

where $S_t$ denotes the realised spread at time $t$. To guard against implausibly large per-trade outcomes, gross P&L is optionally capped as a fraction of the position value $N_t$. This cap bounds the maximum gain or loss per trade to a fixed percentage of allocated notional and, combined with constant-fraction sizing, limits the portfolio impact of extreme observations in a transparent manner. Such constraints reflect the general principle that backtests should encode feasible execution and risk limits rather than assume unbounded scalability (Chan, 2013, pp. 189–194; López de Prado, 2018, pp. 125–130).

Transaction costs are incorporated directly at the trade level and deducted from gross P&L. The implementation employs a fixed cost per MWh traded, so that net P&L is:

$$
\text{P\&L}^{\text{net}}_t = \text{P\&L}^{\text{gross}}_t - q_t \cdot c_{\text{fixed}}
$$

In the baseline specification employed in this study, the fixed transaction cost is set to $c_{\text{fixed}} = 0.06$ EUR/MWh per executed trade, corresponding to 0.03 EUR/MWh for opening and 0.03 EUR/MWh for closing. When no trade is taken ($a_t = 0$) or when the spread is undefined, costs are zero. Portfolio value is updated recursively as:

$$
V_{t+1} = \max(V_t + \text{P\&L}^{\text{net}}_t, 0)
$$

with a non-negativity floor to prevent economically meaningless negative equity in the simulation.

### 3.7.4 Performance Metrics and Trade Logging

In addition to producing an equity curve and period-by-period returns, the framework records a detailed trade log comprising the action taken, prices, spread, traded volume (MWh), gross and net P&L, and whether caps were active. Summary statistics are subsequently computed from the simulated return series, including:

- **Total Return**: Cumulative percentage change in portfolio value
- **Annualised Return**: Time-scaled return assuming continuous compounding
- **Sharpe Ratio**: Risk-adjusted return computed as the ratio of mean excess return to return volatility
- **Sortino Ratio**: Downside risk-adjusted return using only negative return observations in the denominator
- **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value
- **Win Rate**: Proportion of trades yielding positive net P&L

These metrics align with standard performance measures employed in systematic trading evaluation (Chan, 2013, pp. 45–62) and enable direct comparison with the statistical evaluation framework presented in Section 3.6. The emphasis throughout is that the backtest rules are fixed prior to evaluation and applied uniformly across all candidate strategies, enabling a fair out-of-sample comparison that reflects realistic execution constraints (Campbell, Lo, & MacKinlay, 1997, pp. 149–180; López de Prado, 2018, pp. 101–130).

---

## References

Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.

Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. John Wiley & Sons.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
