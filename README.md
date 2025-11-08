# Energy Price Prediction using News Sentiment Analysis

This project predicts German electricity prices using sentiment and topic analysis of news articles combined with historical energy market data. The methodology leverages both English and German news sources, with a focus on energy-related topics, to forecast electricity prices 24 hours ahead.

## Project Overview

The project uses a comprehensive dataset of news articles (English and German) collected over a 5-year period (November 2020 - October 2025) from major news outlets. These articles are processed, cleaned, and analyzed to extract features that are combined with historical energy market data to predict future electricity prices.

## Repository Structure

### Main Directories

- **`data/`**: News data collection scripts, raw datasets, and data cleaning notebooks
  - Contains fetchers for English and German news via NewsAPI.org
  - Includes detailed data analysis notebooks (`data_analysis_v1.ipynb`, `data_analysis_v2.ipynb`, `data_analysis_v3.ipynb`)
  - Final cleaned datasets: `english_news_v9.csv` (~48K articles), `german_news_v1.csv` (~117K articles)
  - See `/data/README.md` for detailed data collection and cleaning methodology

- **`methodology_v3/`**: Current methodology implementation for price prediction
  - Unified approach combining topic-based and embedding-based features from news
  - Uses zero-shot classification and semantic embeddings with time-decayed aggregation
  - Implements expanding window training with Ridge Regression and XGBoost models
  - See `/methodology_v3/README.md` for detailed methodology description

### Legacy Directories

- **`methodology_v1/`**: Earlier iteration using topic-based features only
- **`methodology_v2/`**: Earlier iteration using embedding-based features only
- **`legacy_notebooks/`**: Historical exploration notebooks

## Key Features

- **Multi-source News Collection**: Aggregates news from 11+ English sources and 7+ German sources
- **Rigorous Data Cleaning**: Multi-round topic modeling with BERTopic to remove irrelevant content (sports, entertainment, etc.)
- **Hybrid Feature Engineering**: Combines topic classification and semantic embeddings from news articles
- **Time-Decayed Weighting**: Recent news has more influence on predictions than older news
- **Robust Validation**: Strict train/validation/test split with expanding window approach

## Data Sources

- **NewsAPI.org**: Primary source for both English and German news articles
- **Energy Charts API**: Provides hourly electricity price and power generation data
- **Coverage**: November 1, 2020 - October 31, 2025

## Getting Started

1. **Install dependencies** (requirements not currently tracked, see notebooks for imports)
2. **Data Collection**: Review scripts in `/data` directory
   - `english_news_fetcher.py`: Fetch English news articles
   - `german_news_fetcher.py`: Fetch German news articles
3. **Data Analysis**: Explore data cleaning process in `/data/data_analysis_v*.ipynb`
4. **Model Training**: Run notebooks in `/methodology_v3` to train and evaluate models

## Current Status

The project currently uses Methodology V3, which combines the best aspects of previous approaches. See `/methodology_v3/proposed_optimisations.md` for planned improvements and optimization strategies.
