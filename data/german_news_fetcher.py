"""
Download German news articles from NewsAPI.org.

This script downloads news articles from specified sources using keyword-based queries
and saves them to a CSV file for energy market analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from tqdm import tqdm


def fetch_news_data(start_date: str, end_date: str, language: str, keywords: str, 
                    source: str, api_key: str) -> pd.DataFrame:
    """
    Fetch news articles from NewsAPI.org with flexible keywords.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    language : str
        Language of the news articles
    keywords : str
        Search keywords
    source : str
        Comma-separated list of sources
    api_key : str
        NewsAPI API key
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: 'publishedAt', 'title', 'source', 'description', 'url'
    """
    
    newsapi = NewsApiClient(api_key=api_key)
    
    # Calculate date ranges to fetch in daily increments
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    all_articles = []

    total_days = (end_date - start_date).days
    num_periods = total_days

    # Loop for fetching data in daily increments
    for period in tqdm(range(num_periods), desc="Fetching news data"):
        
        to_date = end_date - timedelta(days=period)
        from_date = to_date - timedelta(days=1)
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')

        try:
            articles_period = newsapi.get_everything(
                q=keywords,
                sources=source,
                language=language,
                sort_by='relevancy',
                from_param=from_str,
                to=to_str,
                page_size=100
            )
            
            if 'articles' in articles_period:
                all_articles.extend(articles_period['articles'])
        except Exception as e:
            print(f"Warning: Error fetching data for {from_str} to {to_str}: {e}")
            continue

    # Conversion to DataFrame
    news_df = pd.DataFrame([
        {
            'publishedAt': article['publishedAt'],
            'title': article.get('title', ''),
            'source': article['source']['name'],
            'description': article.get('description', ''),
            'url': article.get('url', '')
        }
        for article in all_articles
    ])

    # Date Conversion, Sorting and Duplicate Removal + Validation Check
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df = news_df.sort_values('publishedAt')
    news_df = news_df.drop_duplicates(subset=['title'], keep='first')

    if news_df.empty:
        print("Warning: No articles fetched!")
        return pd.DataFrame(columns=['publishedAt', 'title', 'source', 'description', 'url'])
    else:
        print(f"Successfully fetched {len(news_df)} articles")
        return news_df


def main():
    """Main execution function."""
    
    # Define search parameters
    start_date = '2020-11-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Keywords
    keywords = (
        #'Strom OR Energie OR Kraftwerk OR Erneuerbare OR Atomkraft OR Kohlekraftwerk OR Gaskraftwerk OR Windkraft OR '
        #'Solar OR Energiepolitik OR Strommarkt OR Energienetz OR Stromnetz OR Strompreis OR Energiepreis OR Stromverbrauch OR '
        #'Energieverbrauch OR EZB OR Stromrechnung OR Energiekosten OR Energiekrise OR Stromausfall OR Energieversorgung OR '
        #'Heizung OR Klimaanlage OR Kältewelle OR Hitzewelle OR Markt OR Preis OR Politik OR Regulierung OR Stromversorgung'
        'Stromerzeugung OR Stromnachfrage OR Strombedarf OR Stromsparen OR Stromeffizienz OR Stromsparen OR Stromsparen OR Stromsparen OR'
        'Wetter OR EZB OR EPEX OR Zentralbank OR Erdöl OR Erdgas OR Kohle OR OPEC OR IEA OR EU'
   )
    
    # Sources
    sources = (
        'bild, der-tagesspiegel, die-zeit, focus, handelsblatt, spiegel-online, wirtschafts-woche'
    )
    
    # Load API key from environment
    load_dotenv()
    NEWS_API_KEY = os.getenv('NEWSAPIORG_KEY')
    
    if not NEWS_API_KEY:
        raise ValueError("NEWSAPIORG_KEY not found in environment variables. "
                        "Please set it in your .env file.")
    
    print(f"Downloading German news articles from {start_date} to {end_date}")
    print(f"Keywords: {keywords}")
    print(f"Sources: {sources}")
    print("-" * 80)
    
    # Fetch the data
    news_df = fetch_news_data(start_date, end_date, 'de', keywords, sources, NEWS_API_KEY)
    
    # Display summary statistics
    if not news_df.empty:
        print("-" * 80)
        print(f"Total articles: {len(news_df)}")
        print(f"Date range: {news_df['publishedAt'].min()} to {news_df['publishedAt'].max()}")
        print(f"Unique sources: {news_df['source'].nunique()}")
        print(f"\nArticles per source:")
        print(news_df['source'].value_counts())
        
        # Save to CSV
        output_file = 'german_news_raw2.csv'
        news_df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()