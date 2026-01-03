Section 3 text updates after code refactor

Update 1: Section 3.1 Framework Structure (models and pipeline description)
Old text:
```
• Explain and highlight which models will be used (XGBoost, Ridge & Logistic 
Regression). We use a dual model set-up, where we first classify the next day 
spread to be positive or negative alongside a confidence score, then use a 
Logistic (or lightGBM) to then turn those forecast into long/short/hold decisions 
```
New text:
```
• Explain and highlight which models will be used (RidgeCV for time-decay
parameter selection, XGBoost for calibrated multi-class spread classification, and
LightGBM as a meta-learner that combines baseline features with XGBoost
predictions to output Long/Neutral/Short signals). A price-only LightGBM model
serves as the baseline benchmark.
```

Update 2: Section 3.4.2 Topic taxonomy and hierarchical routing
Old text:
```
The topic taxonomy was designed to capture the primary drivers of electricity price 
variation identified in Section 2.1  and was  organised into a two -stage hierarchical 
structure that balances classification precision with computational efficiency. The first 
stage routes articles to one of seven high-level categories, while the second stage assigns 
articles to specific leaf topics within the selected category. 
The seven high-level categories reflect distinct channels through which news content may 
influence electricity prices: 
• Nachfrage (Demand): News affecting electricity consumption levels 
• Angebot (Supply): News concerning generation capacity and availability 
• Brennstoffpreise (Fuel Prices): News on input costs for thermal generation 
• Makrofinanzen (Macro Finance): Broader economic conditions affecting markets 
• Geopolitik (Geopolitics): International developments impacting energy supply 
• Wetter (Weather): Meteorological conditions influencing demand and renewable 
output 
• Sonstiges (Other): Articles lacking clear energy market relevance 
...
Category Leaf Topics 
Nachfrage Electricity consumption in Germany rising; Electricity consumption in Germany 
falling 
Angebot Wind and solar generation rising; Wind and solar generation falling; Grid or plant 
outages reducing supply; Expansion of LNG terminals, pipelines, or plants 
increasing supply 
Brennstoffpreise Wholesale gas prices rising; Wholesale gas prices falling 
Makrofinanzen Rising interest rates or high inflation tightening markets; Falling interest rates or 
easing inflation calming markets 
Geopolitik Geopolitical tensions or sanctions exacerbating energy supply; Geopolitical détente 
or lifted sanctions reducing supply risks 
Wetter Cold, calm, or overcast conditions increasing price pressure; Mild weather, strong 
wind, or sunshine easing prices 
Sonstiges No relevance to energy, weather, or financial markets 
Table XX:  
...
Each article's title is processed against 
the seven category labels using the zero -shot pipeline. The model returns probab ility 
scores reflecting the confidence of each category assignment. If the highest -scoring 
category exceeds the Stage 1 threshold ( τ₁ = 0.35), the article proceeds to Stage 2 
classification within that category; otherwise, it is routed to the fallback cate gory 
("Sonstiges"). 
...
If the selected category contains multiple leaf topics, a second inference pass determines 
the specific topic. A Stage 2 threshold (τ₂ = 0.25) governs whether the leaf assignment is 
retained or whether the article falls back to the default "other" topic. 
...
The threshold values ( τ₁ = 0.35, τ₂ = 0.25) were determined through manual inspection 
of classification score distributions, balancing the trade -off between coverage (retaining 
more articles in specific categories) and precision (avoiding misclassification of 
ambiguous content), following the calibration approach discussed by Laurer et al. (2024). 
```
New text:
```
The topic taxonomy was designed to capture the primary drivers of electricity price 
variation identified in Section 2.1 and was organised into a two-stage hierarchical 
structure that balances classification precision with computational efficiency. The first 
stage routes articles to one of eleven high-level categories, while the second stage assigns 
articles to specific leaf topics within the selected category. 
The eleven high-level categories reflect distinct channels through which news content may 
influence electricity prices: 
• Nachfrage (Stromverbrauch) 
• Angebot (Erzeugung & Infrastruktur) 
• Brennstoffpreise 
• Wetter 
• Wirtschaft & Konjunktur 
• Finanzmärkte & Geldpolitik 
• Handel & Außenwirtschaft 
• Geopolitik & Konflikte 
• Technologie & Industrie 
• Politik & Regulierung 
• Sonstiges 

Table XX (updated): Topic taxonomy
| Category | Leaf Topics |
| --- | --- |
| Nachfrage (Stromverbrauch) | Steigender Stromverbrauch durch Wirtschaft, Industrie oder Extremwetter; Sinkender Stromverbrauch durch Konjunkturschwäche oder mildes Wetter |
| Angebot (Erzeugung & Infrastruktur) | Kraftwerksausfälle, Netzengpässe oder geringe erneuerbare Einspeisung; Hohe erneuerbare Einspeisung, neue Kapazitäten oder stabile Netze |
| Brennstoffpreise | Steigende Gas-, Kohle- oder CO₂-Preise; Fallende Gas-, Kohle- oder CO₂-Preise |
| Wetter | Kaltes, windarmes oder bewölktes Wetter; Mildes Wetter, starke Winde oder viel Sonneneinstrahlung |
| Wirtschaft & Konjunktur | Positive Wirtschaftsentwicklung, steigende Industrieproduktion oder Unternehmenswachstum; Rezession, Unternehmenskrise oder sinkende Industrieproduktion |
| Finanzmärkte & Geldpolitik | Zinsentscheidungen, Inflation oder Währungsschwankungen; Börsennachrichten, Unternehmensgewinne oder Investitionen |
| Handel & Außenwirtschaft | Zölle, Handelskonflikte oder Exportbeschränkungen; Handelsabkommen, Marktöffnung oder Lieferkettenentwicklung |
| Geopolitik & Konflikte | Krieg, Sanktionen, Terrorgefahr oder geopolitische Spannungen; Friedensgespräche, Diplomatie oder Aufhebung von Sanktionen |
| Technologie & Industrie | Technologieentwicklung, Halbleiter, Elektromobilität oder Batterieproduktion; Industriepolitik, Produktionsstandorte oder Unternehmensstrategien |
| Politik & Regulierung | Energiepolitik, Klimagesetze oder EU-Regulierung; Innenpolitik, Regierungsbildung oder Wahlen |
| Sonstiges | Sport, Unterhaltung, Kultur oder Lokalnachrichten ohne Wirtschaftsbezug |

Each article's title is processed against the eleven category labels using the zero-shot 
pipeline. The model returns probability scores reflecting the confidence of each category 
assignment. If the highest-scoring category exceeds the Stage 1 threshold (τ₁ = 0.25), 
the article proceeds to Stage 2 classification within that category; otherwise, it is routed 
to the fallback category ("Sonstiges"). 
...
If the selected category contains multiple leaf topics, a second inference pass determines 
the specific topic. A Stage 2 threshold (τ₂ = 0.20) governs whether the leaf assignment is 
retained or whether the article falls back to the default "other" topic. 
...
The threshold values (τ₁ = 0.25, τ₂ = 0.20) were determined through manual inspection 
of classification score distributions, balancing the trade-off between coverage (retaining 
more articles in specific categories) and precision (avoiding misclassification of 
ambiguous content), following the calibration approach discussed by Laurer et al. (2024). 
```

Update 3: Section 3.4.4 Time-decay topic aggregation dimensions
Old text:
```
For each hourly timestamp in the energy dataset, time -decayed weighted counts are 
computed for each of the 14 leaf topics. The aggregation proceeds as follows: 
...
The output is a matrix of dimension (T × 14), where 𝑇 denotes the number of hourly 
timestamps, and each column represents the time -decayed count for a given topic. 
```
New text:
```
For each hourly timestamp in the energy dataset, time-decayed weighted counts are 
computed for each of the 20 leaf topics (excluding the "Sonstiges" label). The aggregation 
proceeds as follows: 
...
The output is a matrix of dimension (T × 20), where 𝑇 denotes the number of hourly 
timestamps, and each column represents the time-decayed count for a given topic. 
```

Update 4: Section 3.4.5 Dimensionality reduction and feature count
Old text:
```
These 20 embedding dimensions, combined with the 14 topic count features, yield 34 
news-derived features per parameter combination. 
```
New text:
```
These 20 embedding dimensions, combined with the 20 topic count features, yield 40 
news-derived features per parameter combination. 
```

Update 5: Section 3.4.6 Feature integration table and proportion
Old text:
```
Topic Features 14 Time-decayed counts per leaf topic 
Embeddings Features 20 UMAP-reduced embedding dimensions 
...
The news-derived features (34 dimensions) account for approximately 72% of the total 
feature space, reflecting the research focus on extracting predictive signal s from textual 
information. 
```
New text:
```
Topic Features 20 Time-decayed counts per leaf topic 
Embeddings Features 20 UMAP-reduced embedding dimensions 
...
The news-derived features (40 dimensions) account for approximately 75% of the total 
feature space (40 of 53), reflecting the research focus on extracting predictive signals 
from textual information. 
```

Update 6: Section 3.7.1 Trading strategy specification (signal generation)
Old text:
```
1. XGBoost base model generates calibrated probability estimates for spread direction 
2. LightGBM meta-learner refines these into final Long/Neutral/Short signals 
```
New text:
```
1. XGBoost base model generates calibrated three-class probability estimates and a 
predicted class for the spread target (Short/Neutral/Long) 
2. LightGBM meta-learner combines baseline features with the XGBoost prediction and 
probability features to output final Long/Neutral/Short signals 
```
