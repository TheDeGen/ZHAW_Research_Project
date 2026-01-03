# Section 3.4 Feature Engineering — Condensed Rewrite

## Analysis Summary

**Original length estimate:** ~3,500 words (pages 34–43)  
**Target length:** ~2,450 words (~30% reduction)  
**Key reductions:**
- Removed NLI/entailment theory (already in Section 2.2.2)
- Removed DeBERTa architecture details (already in Section 2.2.2)
- Removed MiniLM/knowledge distillation theory (already in Section 2.2.2)
- Removed UMAP theory (already in Section 2.2.2)
- Removed time-decay theoretical motivation (already in Section 2.3)
- Consolidated repetitive procedural descriptions
- Updated parameters to reflect current implementation (11 categories, 21 leaf topics)

---

## Condensed Section 3.4: Feature Engineering

The feature engineering process transforms raw energy market data and preprocessed news articles into a structured feature matrix suitable for predictive modelling. This section details the construction of baseline features, the extraction of topical signals via zero-shot classification, the generation of semantic embeddings, and their temporal aggregation using exponential-decay weighting.

### 3.4.1 Baseline Features

The baseline feature set comprises variables derived directly from the energy market data described in Section 3.2.2, capturing the fundamental supply–demand dynamics and temporal patterns that drive electricity price formation.

**Price and Spread Features:** The primary price variables are the day-ahead auction price and the intraday spot price (both EUR/MWh), resampled at hourly frequency. The spread between these prices—computed as the difference between spot and day-ahead values—serves as the **target variable** for classification. Observations are labelled as **Long** (positive spread, spot > day-ahead), **Short** (negative spread, spot < day-ahead), or **Neutral** when the absolute spread falls within a symmetric deadband of ±3.0 EUR/MWh. This three-class formulation captures the directional trading opportunity whilst accounting for periods where price differences are insufficient to justify position-taking after transaction costs.

**Generation and Load Features:** Aggregate electricity generation and total system load (MW) provide direct measures of supply–demand balance. As discussed in Section 2.1.4, the merit-order effect implies that price formation depends critically on the marginal generating unit dispatched to meet demand.

**Lagged Features:** To capture autocorrelation in electricity prices and demand patterns, lagged values are included at two horizons: 24 hours and 168 hours (one week). Specifically, the feature set contains `price_lag_24`, `price_lag_168`, `load_lag_24`, and `load_lag_168`. These lags reflect the strong daily and weekly seasonality in electricity consumption patterns discussed in Section 2.1.4, where demand exhibits pronounced differences between weekdays and weekends and between daytime peak hours and overnight off-peak periods.

**Temporal Features:** Categorical and cyclical time indicators encode the position of each observation within relevant periodic structures:
- `hour` (0–23): Captures intraday variation
- `day_of_week` (0–6): Distinguishes weekday from weekend dynamics
- `day_of_year` (1–366): Encodes annual seasonal effects
- `week_of_year` (1–52): Provides coarser seasonal resolution
- `month` (1–12): Captures broader seasonal trends

Together, these 13 baseline features establish a foundation that captures the structural characteristics of electricity markets before the incorporation of news-derived signals.

### 3.4.2 Topic Classification via Zero-Shot Learning

News articles are transformed into structured topical features via a zero-shot classification pipeline that assigns each article to predefined energy-relevant categories derived from Section 2.1. This approach leverages the semantic generalisation capabilities of NLI-based models, as discussed in Section 2.2.2, enabling classification without task-specific training data (Yin et al., 2019).

The classification employs the mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 model (Laurer et al., 2024), using the hypothesis template: *"Der Artikel handelt von: {label}."*

**Hierarchical Taxonomy:** The topic structure comprises eleven high-level categories reflecting distinct channels through which news may influence electricity prices. Each category (except "Sonstiges") contains two directionally opposed leaf topics—one representing conditions that would exert upward pressure on prices, and one representing conditions that would ease prices—enabling the model to capture both topical relevance and implied price impact direction:

| Category | Leaf Topics |
|----------|-------------|
| Nachfrage (Stromverbrauch) | Rising consumption (economic activity/extreme weather); Falling consumption (weak economy/mild weather) |
| Angebot (Erzeugung & Infrastruktur) | Plant outages/grid constraints/low renewables; High renewables/new capacity/stable grids |
| Brennstoffpreise | Rising gas, coal, or CO₂ prices; Falling gas, coal, or CO₂ prices |
| Wetter | Cold, calm, or overcast conditions; Mild weather, strong wind, or sunshine |
| Wirtschaft & Konjunktur | Positive economic development; Recession or declining production |
| Finanzmärkte & Geldpolitik | Interest rate decisions/inflation; Stock market news/investments |
| Handel & Außenwirtschaft | Tariffs/trade conflicts; Trade agreements/market opening |
| Geopolitik & Konflikte | War/sanctions/geopolitical tensions; Peace talks/diplomacy |
| Technologie & Industrie | Technology development; Industrial policy |
| Politik & Regulierung | Energy policy/climate legislation; Domestic politics/elections |
| Sonstiges | Sports, entertainment, or local news without economic relevance |

This yields 21 leaf topics across the eleven categories.

**Two-Stage Classification Procedure:** Each article's title is first classified against the eleven category labels. If the highest-scoring category exceeds the Stage 1 threshold (τ₁ = 0.25), the article proceeds to Stage 2 classification within that category; otherwise, it is routed to "Sonstiges." In Stage 2, a threshold of τ₂ = 0.20 governs whether the specific leaf topic is retained. Articles initially classified as "Sonstiges" undergo secondary classification using the article description, providing a fallback for ambiguous headlines.

The threshold values were determined through manual inspection of classification score distributions, balancing coverage against precision (Laurer et al., 2024). Articles ultimately classified as having no energy relevance are excluded from subsequent aggregation.

### 3.4.3 Sentence Embedding Generation

Complementing discrete topic classification, dense sentence embeddings capture semantic information that predefined categories may not fully represent. As introduced in Section 2.2.2, sentence embeddings encode the full semantic content of text as continuous vectors, preserving nuanced information beyond what discrete labels can express.

The embedding pipeline employs the paraphrase-multilingual-MiniLM-L12-v2 model (Reimers & Gurevych, 2019), generating 384-dimensional vector representations optimised for semantic similarity across 50+ languages including German. For each article, the embedding is computed from the title using mean pooling over contextualised token representations, prioritising computational efficiency while capturing primary informational content.

### 3.4.4 Time-Decay Aggregation of News Features

Individual article-level features must be aggregated to the hourly resolution of the energy market data. The aggregation employs exponential time-decay weighting, assigning greater influence to recent articles while retaining diminishing contributions from older content. This reflects the intuition that news impact decays as information is absorbed into prices (Tetlock, 2007; Li et al., 2021).

The exponential decay formulation is:

$$w_i = e^{-\lambda \cdot h_i}$$

where $w_i$ is the weight assigned to article $i$, $\lambda$ is the decay rate, and $h_i$ is the hours elapsed since publication.

**Topic Aggregation:** For each hourly timestamp $t$, time-decayed weighted counts are computed for each of the 20 leaf topics (excluding "Sonstiges"):

1. Define lookback window: identify articles published within $(t - L, t]$
2. Compute decay weights: $w_i = e^{-\lambda \cdot h_i}$ for each article
3. Aggregate by topic: $\text{topic\_count}_k(t) = \sum_{i \in \text{topic } k} w_i$

The output is a matrix of dimension $(T \times 20)$, where each column represents the time-decayed count for a given topic.

**Embedding Aggregation:** Time-decayed embedding aggregation follows a similar procedure using weighted vector averages:

$$\mathbf{e}_{\text{agg}}(t) = \frac{\sum_i w_i \cdot \mathbf{e}_i}{\sum_i w_i}$$

If no articles fall within the lookback window, the aggregated embedding is set to zero. The output is a matrix of dimension $(T \times 384)$.

**Parameter Selection:** The lookback window ($L$) and decay rate ($\lambda$) jointly determine the temporal dynamics of news feature aggregation. Shorter lookback windows with higher decay rates capture rapid, short-term news effects, while longer windows with lower decay rates smooth over transient fluctuations. Given uncertainty regarding optimal values, a grid search across multiple parameter combinations is employed, with the optimal combination selected through the model validation procedure described in Section 3.5.

### 3.4.5 Dimensionality Reduction of Aggregated Embeddings

The 384-dimensional time-decayed embedding vectors pose challenges for downstream modelling due to high dimensionality and overfitting risk. UMAP, introduced in Section 2.2.2, is employed for dimensionality reduction, offering superior scalability and global structure preservation compared to alternatives (McInnes et al., 2020).

The UMAP configuration employs:
- `n_components` = 20: reducing dimensionality by a factor of approximately 19
- `n_neighbors` = 15: balancing local and global structure preservation
- `min_dist` = 0.1: controlling clustering tightness

The output is a matrix of dimension $(T \times 20)$. These 20 embedding dimensions, combined with the 20 topic count features, yield **40 news-derived features** per parameter combination.

### 3.4.6 Feature Integration

The complete feature matrix integrates baseline features with news-derived features:

| Feature Category | Count | Description |
|------------------|-------|-------------|
| Price Features | 2 | Day-ahead price, spot price |
| Spread Features | 1 | Absolute spread |
| Load Features | 1 | System load |
| Lagged Features | 4 | 24h and 168h lags for price and load |
| Temporal Features | 5 | Hour, day of week, day/week/month of year |
| Topic Features | 20 | Time-decayed counts per leaf topic |
| Embedding Features | 20 | UMAP-reduced embedding dimensions |
| **Total** | **53** | |

The news-derived features (40 dimensions) account for approximately 75% of the total feature space, reflecting the research focus on extracting predictive signals from textual information.

**Dataset Partitioning:** The dataset is partitioned chronologically:
- **Training Set:** First 70% of observations
- **Validation Set:** Next 20% of observations
- **Test Set:** Final 10% of observations

This temporal split ensures test observations occur strictly after training observations, reflecting the realistic constraint that forecasting models cannot access future information.

---

## Change Log

| Subsection | Original | Revised | Key Changes |
|------------|----------|---------|-------------|
| 3.4 Intro | ~100 words | ~50 words | Removed redundant pipeline overview |
| 3.4.1 Baseline | ~350 words | ~320 words | Added target variable definition (Long/Short/Neutral with ±3 EUR/MWh deadband); referenced Section 2.1.4 for seasonality |
| 3.4.2 Topic Classification | ~1,100 words | ~480 words | Removed NLI theory (→ Section 2.2.2), removed DeBERTa architecture details, updated to 11 categories/21 topics, added note on positive/negative leaf pairing |
| 3.4.3 Embeddings | ~400 words | ~100 words | Simplified to reference Section 2.2.2, kept only model choice and practical details |
| 3.4.4 Time-Decay | ~800 words | ~380 words | Removed theoretical motivation (→ Section 2.3), removed specific parameter grid values (deferred to Section 3.5) |
| 3.4.5 Dimensionality | ~350 words | ~150 words | Removed UMAP theory (→ Section 2.2.2), updated feature counts |
| 3.4.6 Integration | ~400 words | ~200 words | Removed normalisation/StandardScaler section, renamed to "Feature Integration" |

**Total estimated reduction:** ~35% (from ~3,500 to ~2,250 words)

## Notes for Final Document

1. **Cross-references added:**
   - 3.4.1: Section 2.1.4 for seasonality patterns in electricity demand
   - 3.4.2: Section 2.2.2 for NLI-based classification
   - 3.4.3: Section 2.2.2 for sentence embeddings introduction
   - 3.4.4: Section 3.5 for parameter selection details
   - 3.4.5: Section 2.2.2 for UMAP

2. **Key clarifications added:**
   - Target variable is the spread, discretised into Long/Short/Neutral using ±3.0 EUR/MWh deadband
   - Each category (except Sonstiges) has two directionally opposed leaf topics (price-increasing vs price-decreasing)

3. **Removed:**
   - Normalisation/StandardScaler section
   - Specific grid search parameter values (now deferred to Section 3.5)

4. **Parameters reflect current implementation:**
   - Stage 1 threshold: 0.25
   - Stage 2 threshold: 0.20
   - High-level categories: 11
   - Leaf topics: 20 (excluding Sonstiges)
   - Total news features: 40
   - Total features: 53
