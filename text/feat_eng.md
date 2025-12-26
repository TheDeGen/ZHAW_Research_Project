# 3.4 Feature Engineering

The feature engineering process transforms raw energy market data and preprocessed news articles into a structured feature matrix suitable for predictive modelling. This section details the construction of baseline features derived from electricity market fundamentals, the extraction of topical signals through zero-shot classification, the generation of semantic embeddings from news text, and the temporal aggregation of these news-derived features using exponential decay weighting. The resulting feature set integrates both quantitative market dynamics and qualitative information encoded in news discourse, thereby enabling the model to capture the complex interplay between market fundamentals and information flows that characterise short-term electricity price movements.

## 3.4.1 Baseline Features

The baseline feature set comprises variables derived directly from the energy market data described in Section 3.3.2. These features capture the fundamental supply-demand dynamics that drive electricity price formation, as well as temporal patterns that reflect the cyclical nature of electricity consumption.

**Price and Spread Features.** The primary price variables include the day-ahead auction price (EUR/MWh) and the intraday spot price (EUR/MWh), both resampled to hourly frequency. The absolute spread between these prices, computed as the difference between spot and day-ahead values, serves as the basis for the target variable. This spread reflects the deviation between anticipated and realised market conditions, capturing the information advantage that accrues between the day-ahead auction close and real-time delivery.

**Generation and Load Features.** Aggregate electricity generation and total system load (MW) provide direct measures of supply-demand balance. As discussed in Section 2.1.4, the merit-order effect implies that price formation depends critically on the marginal generating unit dispatched to meet demand. Aggregate load serves as a proxy for demand pressure, where higher consumption during peak periods typically elevates prices by requiring dispatch of higher-cost thermal generation (Edwards, 2023, p. 135).

**Lagged Features.** To capture autocorrelation structures inherent in electricity prices and demand patterns, lagged values of key variables are incorporated at two horizons: 24 hours (daily cycle) and 168 hours (weekly cycle). Specifically, the feature set includes `price_lag_24`, `price_lag_168`, `load_lag_24`, and `load_lag_168`. These lags reflect the strong diurnal and weekly seasonality documented in electricity market research, where demand patterns exhibit pronounced differences between weekdays and weekends, and between daytime peak hours and overnight off-peak periods (Weron, 2014, p. 1035).

**Temporal Features.** Categorical and cyclical time indicators encode the position of each observation within relevant periodic structures:
- `hour` (0–23): Captures intraday variation aligned with consumption patterns
- `day_of_week` (0–6): Distinguishes weekday from weekend dynamics
- `day_of_year` (1–365): Encodes annual seasonal effects, including heating and cooling demand
- `week_of_year` (1–52): Provides coarser seasonal resolution
- `month` (1–12): Captures broader seasonal trends in renewable generation and demand

Together, these 13 baseline features establish a foundation that reflects the structural characteristics of electricity markets prior to the incorporation of news-derived signals.

## 3.4.2 Topic Classification via Zero-Shot Learning

The transformation of news articles into structured topical features proceeds through a zero-shot classification pipeline that assigns each article to predefined energy-relevant categories without requiring task-specific training data. This approach leverages the semantic generalisation capabilities of large language models, enabling classification across diverse topic labels based solely on natural language descriptions (Yin et al., 2019, pp. 1–2).

**Model Selection.** The classification pipeline employs the `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` model developed by Laurer et al. (2022, 2023), which extends Microsoft's DeBERTa-v3 architecture (He et al., 2023) to multilingual zero-shot classification. DeBERTa-v3 improves upon earlier transformer architectures through disentangled attention mechanisms that separately encode content and position information, combined with ELECTRA-style replaced token detection during pre-training, achieving state-of-the-art performance on natural language understanding benchmarks (He et al., 2023, pp. 1–2).

The selected model was fine-tuned on the Cross-Lingual Natural Language Inference (XNLI) dataset and the multilingual-NLI-26lang-2mil7 corpus, comprising over 2.7 million hypothesis-premise pairs across 27 languages. This training enables the model to perform natural language inference across 100 languages, achieving accuracy scores between 79.4% and 87.1% on the XNLI test set (Laurer et al., 2023). The multilingual capability is essential for processing German-language news content whilst benefiting from the broader linguistic patterns learned during pre-training on diverse language data.

Zero-shot classification through NLI operates by reformulating the classification task as textual entailment (Yin et al., 2019). For each candidate label, the model evaluates whether a hypothesis constructed from the label is entailed by the input text (premise). The hypothesis template employed is:

> "Der Artikel handelt von: {label}."
> (English: "The article is about: {label}.")

The model outputs entailment probabilities for each label, with the highest-probability label assigned as the classification. This formulation enables flexible classification across arbitrary label sets without retraining, a property particularly valuable when domain-specific labelled data are scarce or costly to obtain (Laurer et al., 2022, p. 2).

**Hierarchical Classification Taxonomy.** The topic taxonomy was designed to capture the primary drivers of electricity price variation identified in Section 2.1.4, organised into a two-stage hierarchical structure that balances classification precision with computational efficiency. The first stage routes articles to one of seven high-level categories, while the second stage assigns articles to specific leaf topics within the selected category.

The seven high-level categories reflect distinct channels through which news content may influence electricity prices:

1. **Nachfrage (Demand)**: News affecting electricity consumption levels
2. **Angebot (Supply)**: News concerning generation capacity and availability
3. **Brennstoffpreise (Fuel Prices)**: News on input costs for thermal generation
4. **Makrofinanzen (Macro Finance)**: Broader economic conditions affecting markets
5. **Geopolitik (Geopolitics)**: International developments impacting energy supply
6. **Wetter (Weather)**: Meteorological conditions influencing demand and renewable output
7. **Sonstiges (Other)**: Articles lacking clear energy market relevance

This categorisation draws on the supply and demand factors discussed in Section 2.1.4, where fuel prices, weather conditions, and geopolitical developments were identified as key determinants of electricity price dynamics. Within each high-level category, more granular leaf topics capture directional implications for electricity prices. The complete taxonomy comprises 14 leaf topics:

| Category | Leaf Topics |
|----------|-------------|
| Nachfrage | Electricity consumption in Germany rising; Electricity consumption in Germany falling |
| Angebot | Wind and solar generation rising; Wind and solar generation falling; Grid or plant outages reducing supply; Expansion of LNG terminals, pipelines, or plants increasing supply |
| Brennstoffpreise | Wholesale gas prices rising; Wholesale gas prices falling |
| Makrofinanzen | Rising interest rates or high inflation tightening markets; Falling interest rates or easing inflation calming markets |
| Geopolitik | Geopolitical tensions or sanctions exacerbating energy supply; Geopolitical détente or lifted sanctions reducing supply risks |
| Wetter | Cold, calm, or overcast conditions increasing price pressure; Mild weather, strong wind, or sunshine easing prices |
| Sonstiges | No relevance to energy, weather, or financial markets |

The directional framing of leaf topics (rising/falling, increasing/decreasing) enables the model to capture not merely topical relevance but also the implied price impact direction, which is critical for generating actionable trading signals. It should be noted that certain factors relevant to electricity price formation, such as carbon prices under the EU Emissions Trading System (EU ETS) and cross-border interconnector flows, are not explicitly represented in this taxonomy. These omissions reflect the focus on factors most directly captured in German-language news discourse and represent a limitation of the current approach.

**Routing Logic and Confidence Thresholds.** The hierarchical classification proceeds through two inference stages with confidence-based fallback logic to handle ambiguous cases. Each article's title is processed against the seven category labels using the zero-shot pipeline. The model returns probability scores reflecting the confidence of each category assignment. If the highest-scoring category exceeds the Stage 1 threshold (τ₁ = 0.35), the article proceeds to Stage 2 classification within that category; otherwise, it is routed to the fallback category ("Sonstiges").

Within the assigned category, the article is classified against the corresponding leaf topics. If the selected category contains multiple leaf topics, a second inference pass determines the specific topic. A Stage 2 threshold (τ₂ = 0.25) governs whether the leaf assignment is retained or whether the article falls back to the default "other" topic.

Articles initially classified as "Sonstiges" (no energy relevance) undergo a secondary classification using the article description rather than the title. This fallback addresses cases where headlines are ambiguous but article summaries provide clearer topical signals. The reclassified results replace the initial "other" assignment where a more specific topic achieves sufficient confidence.

The threshold values (τ₁ = 0.35, τ₂ = 0.25) were determined through manual inspection of classification score distributions, balancing the trade-off between coverage (retaining more articles in specific categories) and precision (avoiding misclassification of ambiguous content), following the calibration approach discussed by Laurer et al. (2022). Lower Stage 2 thresholds reflect the observation that within-category classification tends to produce lower absolute confidence scores due to the semantic similarity among leaf topics within the same category.

**Classification Distribution.** The zero-shot classification pipeline was applied to the preprocessed news corpus of 149,512 articles. Table X presents the distribution of articles across leaf topics, illustrating the topical composition of the dataset.

[TABLE: Distribution of articles across 14 leaf topics - to be populated with actual results]

Articles classified as having no energy relevance are excluded from subsequent time-decay aggregation, ensuring that the news-derived features reflect only content with plausible market relevance. This filtering step is essential given the broad keyword-based collection strategy employed during data sourcing, which inevitably captured some off-topic content despite the preprocessing refinements described in Section 3.3.1.

## 3.4.3 Sentence Embedding Generation

Complementing the discrete topical classification, dense sentence embeddings capture semantic information that may not be fully represented by predefined topic categories. While zero-shot classification assigns articles to discrete labels, embeddings encode the full semantic content of article text as continuous vectors, preserving nuanced information that could inform price predictions.

**Model Architecture.** The embedding pipeline employs the `paraphrase-multilingual-MiniLM-L12-v2` model from the Sentence Transformers library (Reimers & Gurevych, 2019). This model generates 384-dimensional dense vector representations optimised for semantic similarity tasks across 50+ languages including German.

The architecture builds upon the MiniLM knowledge distillation framework (Wang et al., 2020), which compresses larger transformer models while preserving most of their representational capacity. MiniLM achieves this compression by training a smaller student model to mimic the self-attention distributions of a larger teacher model, yielding compact models suitable for large-scale text processing without substantial performance degradation.

The multilingual capability derives from the training procedure described by Reimers and Gurevych (2020), which extends monolingual sentence embeddings to multiple languages through knowledge distillation on parallel corpora. A fixed English teacher model produces target embeddings, while the student model learns to map both English text and its translations to the same vector representation. This approach ensures that semantically equivalent content in different languages maps to similar positions in the embedding space.

**Embedding Computation.** For each news article, the embedding is computed from the article title using mean pooling over the contextualised token representations produced by the final transformer layer. Titles were selected over full article text for computational efficiency and because headlines typically convey the primary informational content relevant to rapid market reactions. However, this approach may lose nuanced information contained in article bodies, representing a trade-off between computational tractability and information completeness.

The output of this stage is a 384-dimensional embedding vector for each article, stored alongside the article metadata for subsequent temporal aggregation.

## 3.4.4 Time-Decay Aggregation of News Features

Individual article-level classifications and embeddings must be aggregated to the hourly resolution of the energy market data to enable joint modelling. The aggregation procedure employs exponential time-decay weighting, which assigns greater influence to more recent articles while retaining diminishing contributions from older content within a lookback window. This approach reflects the intuition that news impact on market expectations decays over time as information is absorbed and superseded by subsequent developments.

**Theoretical Motivation.** The application of exponential decay weighting to news signals draws on established practices in financial forecasting and time-series analysis. Tetlock (2007) demonstrated that media content influences investor sentiment and stock returns, with the effect diminishing over subsequent trading days as information is incorporated into prices. Taylor (2008) proposed exponentially weighted information criteria for model selection, arguing that recent observations should receive greater weight to reflect current forecast accuracy. In commodity markets, Li and Wang (2021) demonstrated that incorporating time-decayed news sentiment significantly improves crude oil price forecasting, with the decay parameter governing the effective memory horizon of the model. Gianfreda and Grossi (2012) showed that forecast errors in electricity markets exhibit time-varying patterns that can be modelled through weighted aggregation schemes.

Within electricity markets specifically, the high-frequency nature of price dynamics and the rapid incorporation of information suggest that news relevance diminishes relatively quickly. Forecast revisions for renewable generation, updates to plant availability, and shifts in demand expectations are continuously reflected in intraday trading, implying that older news articles provide progressively less marginal information (Weron, 2014, p. 1031).

The exponential decay formulation provides a parsimonious representation of this information decay:

$$w_i = e^{-\lambda \cdot h_i}$$

where $w_i$ is the weight assigned to article $i$, $\lambda$ is the decay rate parameter, and $h_i$ is the number of hours elapsed since article $i$ was published. Higher values of $\lambda$ produce faster decay, concentrating influence on very recent articles, while lower values extend the effective memory horizon, allowing older articles to retain meaningful weight.

**Topic Count Aggregation.** For each hourly timestamp in the energy dataset, time-decayed weighted counts are computed for each of the 14 leaf topics. The aggregation proceeds as follows:

1. **Define Lookback Window.** For timestamp $t$, identify all articles published within the interval $(t - L, t]$, where $L$ is the lookback window in hours.

2. **Compute Decay Weights.** For each article $i$ in the window, calculate the decay weight $w_i = e^{-\lambda \cdot h_i}$, where $h_i = t - t_i$ is the hours since publication.

3. **Aggregate by Topic.** For each topic $k$, compute the weighted count:
$$\text{topic\_count}_k(t) = \sum_{i \in \text{topic } k} w_i$$

4. **Filter Irrelevant Articles.** Articles classified as "Sonstiges" (no energy relevance) are excluded from the aggregation, ensuring that topic features reflect only energy-relevant news content.

The output is a matrix of dimension $(T \times 14)$, where $T$ is the number of hourly timestamps and each column represents the time-decayed count for one topic. This aggregation preserves the interpretability of topical features: higher values indicate greater recent news volume on that topic, weighted by recency. The directional framing of topics (e.g., "gas prices rising" vs. "gas prices falling") enables the model to distinguish between news flows with opposing price implications.

**Embedding Aggregation.** Time-decayed aggregation of embeddings follows a similar procedure, with the key difference that vector averages replace scalar sums:

1. **Define Lookback Window.** Identify articles within $(t - L, t]$ for each timestamp $t$.

2. **Compute Decay Weights.** Calculate $w_i = e^{-\lambda \cdot h_i}$ for each article.

3. **Weighted Average Embedding.** Compute the aggregated embedding as:
$$\mathbf{e}_{\text{agg}}(t) = \frac{\sum_i w_i \cdot \mathbf{e}_i}{\sum_i w_i}$$

where $\mathbf{e}_i$ is the 384-dimensional embedding vector for article $i$.

The weighted average ensures that the aggregated embedding lies within the convex hull of individual article embeddings, preserving semantic interpretability. If no articles fall within the lookback window for a given timestamp, the aggregated embedding is set to zero, indicating the absence of relevant news content. The output is a matrix of dimension $(T \times 384)$, representing the time-decayed average semantic content of recent news at each timestamp.

**Parameter Selection.** The lookback window ($L$) and decay rate ($\lambda$) jointly determine the temporal dynamics of news feature aggregation. Shorter lookback windows with higher decay rates capture rapid, short-term news effects, while longer windows with lower decay rates smooth over transient fluctuations to capture more persistent trends.

Given the uncertainty regarding optimal parameter values, a grid search approach is employed across a range of parameter combinations:

| Parameter | Values Tested |
|-----------|---------------|
| Lookback Window ($L$) | 24, 48, 72, 168, 336, 504 hours |
| Decay Lambda ($\lambda$) | 0.01, 0.05, 0.1, 0.25, 0.5 |

This yields 30 parameter combinations, each producing a distinct set of time-decayed topic and embedding features. The optimal combination is selected through the model validation procedure described in Section 3.5, which evaluates downstream prediction performance across the parameter grid.

The default parameters ($L = 336$ hours, $\lambda = 0.05$) reflect a two-week lookback with moderate decay, balancing responsiveness to recent news against stability from a broader temporal context. With $\lambda = 0.05$, an article published 24 hours ago retains approximately 30% of its initial weight ($e^{-0.05 \times 24} \approx 0.30$), while an article from one week ago (168 hours) retains only about 0.02% ($e^{-0.05 \times 168} \approx 0.0002$).

## 3.4.5 Dimensionality Reduction of Aggregated Embeddings

The 384-dimensional time-decayed embedding vectors present challenges for downstream modelling due to the curse of dimensionality and potential overfitting, particularly given the limited sample size relative to feature dimensionality. Dimensionality reduction techniques project these high-dimensional representations into a lower-dimensional space while preserving essential structure, thereby improving computational efficiency and model generalisation.

The Uniform Manifold Approximation and Projection (UMAP) algorithm (McInnes et al., 2018) serves as the primary dimensionality reduction method. UMAP constructs a topological representation of the high-dimensional data based on fuzzy simplicial sets, then optimises a low-dimensional embedding that preserves this structure. Compared to alternatives such as t-SNE, UMAP offers superior scalability to large datasets and better preservation of global structure, making it well-suited for the continuous semantic spaces produced by transformer embeddings (McInnes et al., 2018, pp. 1–2).

The UMAP configuration employs the following parameters:
- **n_components = 20**: Target dimensionality, reducing the 384-dimensional input by a factor of approximately 19
- **n_neighbors = 15**: Local neighbourhood size, controlling the balance between local and global structure preservation
- **min_dist = 0.1**: Minimum distance between points in the embedding, governing clustering tightness

These parameters follow the guidelines provided by McInnes et al. (2018), where `n_neighbors` values between 10 and 50 are recommended for balancing local and global structure preservation, and `min_dist` values between 0.0 and 0.5 control the tightness of the resulting clusters. The choice of 20 components provides sufficient expressiveness for downstream prediction while substantially reducing feature space dimensionality.

PCA provides a linear alternative that guarantees orthogonal components ordered by variance explained, though it cannot capture the nonlinear manifold structure that UMAP preserves. Empirical comparisons on text embedding tasks suggest that UMAP typically outperforms PCA for preserving semantic clustering structure, though the advantage diminishes for downstream supervised learning tasks where predictive features may not align with variance-maximising directions (Grootendorst, 2022, p. 2).

The final output is a matrix of dimension $(T \times 20)$, where each row represents the reduced embedding for one hourly timestamp. These 20 embedding dimensions, combined with the 14 topic count features, yield 34 news-derived features per parameter combination.

## 3.4.6 Feature Integration and Scaling

The complete feature matrix integrates baseline features with news-derived features, producing the input representation for predictive modelling.

**Feature Summary.** Table X summarises the full feature set:

| Feature Category | Count | Description |
|------------------|-------|-------------|
| Price Features | 2 | Day-ahead price, spot price |
| Spread Features | 1 | Absolute spread |
| Load Features | 1 | System load |
| Lagged Features | 4 | 24h and 168h lags for price and load |
| Temporal Features | 5 | Hour, day of week, day/week/month of year |
| Topic Features | 14 | Time-decayed counts per leaf topic |
| Embedding Features | 20 | UMAP-reduced embedding dimensions |
| **Total** | **47** | |

The news-derived features (34 dimensions) constitute approximately 72% of the total feature space, reflecting the research focus on extracting predictive signal from textual information.

**Standardisation.** News-derived features exhibit different scales and distributions compared to baseline features. To prevent scale-dependent biases in model training, StandardScaler normalisation is applied to the news features:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed from the training set. Critically, scaling parameters are fitted exclusively on training data and subsequently applied to validation and test sets, preventing information leakage from future observations into the feature normalisation.

**Temporal Split Strategy.** The dataset is partitioned chronologically to preserve the temporal ordering essential for time-series forecasting:
- **Training Set**: First 70% of observations
- **Validation Set**: Next 20% of observations
- **Test Set**: Final 10% of observations

This temporal split ensures that all test set observations occur strictly after all training observations, reflecting the realistic constraint that forecasting models cannot access future information. The validation set serves for hyperparameter tuning and time-decay parameter selection, while the test set provides an unbiased estimate of out-of-sample performance.

## 3.4.7 Summary

The feature engineering pipeline transforms raw market data and preprocessed news articles into a 47-dimensional feature representation combining:
1. **Baseline features** capturing electricity market fundamentals and temporal patterns
2. **Topic features** encoding the thematic content of recent news via zero-shot classification
3. **Embedding features** capturing semantic information through transformer-based representations

Time-decay aggregation with exponential weighting bridges the article-level news signals to hourly market timestamps, with parameter selection deferred to the model training phase described in Section 3.5. The resulting feature matrix provides a comprehensive representation of both quantitative market dynamics and qualitative information flows, enabling the subsequent modelling stages to exploit the predictive content embedded in energy market news.

---

## References (to be integrated into main bibliography)

Gianfreda, A., & Grossi, L. (2012). Forecasting Italian electricity zonal prices with exogenous variables. *Energy Economics*, 34(6), 2228-2239.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In *Proceedings of ICLR 2023*. https://arxiv.org/abs/2111.09543

Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2022). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis*, 1-17.

Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2023). Building efficient universal classifiers with natural language inference. *arXiv preprint arXiv:2312.17543*.

Li, X., & Wang, J. (2021). The role of news sentiment in oil futures returns and volatility forecasting: Data-decomposition based deep learning approach. *Energy Economics*, 95, 105140.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*. https://arxiv.org/abs/1908.10084

Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. In *Proceedings of EMNLP 2020*.

Taylor, J. W. (2008). Exponentially weighted information criteria for selecting among forecasting models. *International Journal of Forecasting*, 24(3), 513-524.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139-1168.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *Proceedings of NeurIPS 2020*.

Wang, Z., Yang, J., & Li, Q. (2023). Zero-shot text classification for financial sentiment analysis. *Expert Systems with Applications*, 223, 119861.

Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. *International Journal of Forecasting*, 30(4), 1030-1081.

Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. In *Proceedings of EMNLP-IJCNLP 2019*. https://arxiv.org/abs/1909.00161
