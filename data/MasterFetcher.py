"""
Download English news articles from NewsAPI.org.

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
        DataFrame with columns: 'publishedAt', 'title', 'source', 'description', 'content','url'
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
    start_date = '2020-12-09'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Keywords
    keywords = (
    'exploration OR upstream OR midstream OR downstream OR "E&P" OR "offshore drilling"' 
    'OR "deepwater" OR "oilfield services" OR "seismic surveys" OR "well completions"' 
    'OR "production guidance" OR "Brent-WTI spread" OR "physical premiums" OR differentials' 
    'OR "Dubai crude" OR "Oman crude" OR "jet fuel demand" OR "gasoil market" OR arbitrage'
    )
    
    # Sources
    sources = (
        'abc-news,associated-press,bbc-news,bloomberg,cnn,msnbc,nbc-news,'
        'reuters,the-wall-street-journal,the-washington-post,time,usa-today'
    )
    
    # Load API key from environment
    load_dotenv()
    NEWS_API_KEY = os.getenv('NEWSAPIORG_KEY')
    
    if not NEWS_API_KEY:
        raise ValueError("NEWSAPIORG_KEY not found in environment variables. "
                        "Please set it in your .env file.")
    
    print(f"Downloading English news articles from {start_date} to {end_date}")
    print(f"Keywords: {keywords}")
    print(f"Sources: {sources}")
    print("-" * 80)
    
    # Fetch the data
    news_df = fetch_news_data(start_date, end_date, 'en', keywords, sources, NEWS_API_KEY)
    
    # Display summary statistics
    if not news_df.empty:
        print("-" * 80)
        print(f"Total articles: {len(news_df)}")
        print(f"Date range: {news_df['publishedAt'].min()} to {news_df['publishedAt'].max()}")
        print(f"Unique sources: {news_df['source'].nunique()}")
        print(f"\nArticles per source:")
        print(news_df['source'].value_counts())
        
        # Save to CSV
        output_file = 'MT_news_batch_5.csv'
        news_df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()

    #used keywords 1 Core Terms
    """ 
    'oil OR crude OR Brent OR WTI OR "West Texas Intermediate" '
    'OR "natural gas" OR gas OR LNG OR "liquefied natural gas" OR "pipeline gas" '
    'OR "Henry Hub" OR TTF OR NBP '
    'OR "front-month futures" OR "energy futures" '
    'OR "spot price" AND energy' """

    #used keywords 2 Macro Terms
    """
    '(inflation OR recession OR "interest rates" OR tightening OR "monetary policy" OR "quantitative easing" OR "quantitative tightening" OR "business cycle") '
    'OR ("Federal Reserve" OR Fed OR ECB OR BOE OR BOJ OR PBOC) '
    'OR (GDP OR PMI OR manufacturing OR inventories) '
    'OR (futures OR spot OR "forward curve" OR hedging OR volatility OR contango OR backwardation OR "term structure") '
    'OR (commodity OR "energy markets" OR "supply and demand") '
    'OR (CPI OR PPI OR NFP OR payrolls OR unemployment)' """

    #used keywords 3 geopolitics
    """ 
    '(OPEC OR "OPEC+" OR cartel OR quotas OR "production cuts") '
    'OR ("Middle East" OR Gulf OR "Saudi Arabia" OR UAE OR Qatar OR Kuwait) '
    'OR (Iran OR Iraq OR Russia OR Ukraine OR Venezuela OR Libya) '
    'OR (sanctions OR embargo OR geopolitics OR "energy security") '
    'OR ("Strait of Hormuz" OR "Suez Canal" OR "Black Sea" OR "Bab el-Mandeb")'"""

    #used keywords 4 infrastructure
    """
    '(pipeline OR "gas pipeline" OR "oil pipeline" OR outage OR disruption OR flows) '
    'OR ("Nord Stream" OR Yamal OR "Power of Siberia" OR Druzhba) '
    'OR (terminal OR "LNG terminal" OR regasification OR liquefaction OR FSRU) '
    'OR (storage OR "gas storage" OR injection OR withdrawal OR caverns) '
    'OR (refinery OR refineries OR "refinery outage" OR throughput OR maintenance)' """

    #used keywords 5 supply chain
    """"
    '(tanker OR shipping OR "freight rates" OR "charter rates") '
    'OR (VLCC OR "LNG carrier" OR "oil tanker") '
    'OR ("shipping lanes" OR "maritime traffic")' """

    #used keywords 6 major companies https://companiesmarketcap.com/oil-gas/largest-oil-and-gas-companies-by-market-cap/#google_vignette
    """
    '(Exxon OR Chevron OR Shell OR BP OR TotalEnergies OR Equinor) '
    'OR (Aramco OR Gazprom OR Petrobras OR PetroChina OR CNOOC) '
    'OR (Enbridge OR Valero OR Pioneer OR Glencore OR Trafigura) '
    'AND (oil OR gas OR LNG OR pipeline OR production OR refinery OR energy)' """

    #used keywords 7 batch 1 Weather, Seasonality, Demand & Operational Factors
    """ 
    '"cold weather" OR heatwave OR winter OR "winter demand"' 
    'OR storm OR hurricane OR "Gulf of Mexico" OR outage '
    'OR "El Nino" OR "La Nina" OR "weather forecast"' 
    'OR "power generation" OR "electricity demand" OR "industrial demand" '
    'OR "grid demand" OR "power shortage"' 
    'OR "rig count" OR "drilling activity" OR "shale production"' 
    'OR fracking OR "wellhead production"' """

    #used keywords 8 batch 2 energy Transition, Carbon Policy & LNG Trade Structure
    """
    '"energy transition" OR "carbon tax" OR "carbon market"'
    'OR "emissions trading" OR ETS OR "climate policy"' 
    'OR "renewable energy targets"' 
    'OR "long-term LNG contract" OR "spot LNG" OR "LNG supply"' 
    'OR "LNG demand" OR "cargo cancellations" OR "cargo diversions"' """

    #used keywords 9 batch 3 Oil & Gas Trading / Physical Market Jargon
    """
    'crack spread OR "diesel crack" OR "gasoline crack" OR "3-2-1 spread"'
    'OR refining OR "refining margin" OR "run cuts" OR "utilization rate"' 
    'OR "offtake agreement" OR "lifting schedule" OR "cargo loading"'
    'OR "force majeure" OR "barrels per day" OR bpd OR "production outages"' 
    'OR "unplanned outage" OR "market tightness" OR oversupply OR "LNG trains"' 
    'OR "take-or-pay" OR "contracted volumes" OR "boil-off gas" OR BOG' 
    'OR FLNG OR "gas-to-power"'
    """

    #used keywords 10 batch 4 electricity, Power Markets & Trading Structure Jargon
    """
    '"day-ahead market" OR DAM OR "real-time market" OR RTM OR "balancing market"'
    'OR "capacity auction" OR "capacity market" OR "merit order" OR "marginal plant"' 
    'OR "marginal generator" OR "spark spread" OR "clean spark spread" OR "peak load"' 
    'OR baseload OR dispatch OR curtailment OR "floating storage"' 
    'OR demurrage OR "freight congestion" OR "roll yield"' 
    'OR "open interest" OR "speculative positioning"' 
    'OR "managed money" OR "CTA flows"'
    """

    #used keywords 11 batch 5 oil & gas vocabulary
    """
    'exploration OR upstream OR midstream OR downstream OR "E&P" OR "offshore drilling"' 
    'OR "deepwater" OR "oilfield services" OR "seismic surveys" OR "well completions"' 
    'OR "production guidance" OR "Brent-WTI spread" OR "physical premiums" OR differentials' 
    'OR "Dubai crude" OR "Oman crude" OR "jet fuel demand" OR "gasoil market" OR arbitrage'
    """

    # Headlines for FX
    # Keyword 1 - Core FX and Market Terms
    """
    ("EUR/USD" OR "euro-dollar" OR "euro vs dollar" OR "euro versus dollar")
    OR (EUR AND USD AND ("exchange rate" OR "currency pair" OR "forex" OR FX))
    OR ("dollar index" OR DXY OR "US dollar strength" OR "dollar weakness")
    OR ("euro strength" OR "euro weakness" OR "single currency")
    """

   # Keyword 2 - Monetary Policy & Central Banks
    """
    '"Federal Reserve" OR Fed OR FOMC OR Powell OR '
    '"European Central Bank" OR ECB OR Lagarde OR '
    '"interest rates" OR "rate hike" OR "rate cut" OR tightening OR easing OR '
    '"monetary policy" OR "policy divergence" OR "policy meeting" OR '
    '"balance sheet" OR "quantitative tightening" OR "quantitative easing"'
    """ 

    # Keyword 3 - Inflation, Growth & Macro Data
    """
    'inflation OR CPI OR PPI OR "consumer prices" OR "producer prices" OR "consumer price index" OR '
    'GDP OR "gross domestic product" OR "economic growth" OR "recession" OR slowdown OR "economic outlook" OR '
    'PMI OR ISM OR "business confidence" OR "manufacturing index" OR '
    '"unemployment" OR jobs OR payrolls OR "labor market" OR NFP OR "non farm payrolls" OR '
    '"retail sales" OR "trade balance" OR "current account" OR "Economic Surprise Index" OR "CESI" OR "budget deficit"'
    """ 

    # Keyword 4 - Fiscal & Political Developments
    """
    '"US government shutdown" OR "US debt ceiling" OR "Treasury yields" OR '
    '"EU fiscal rules" OR "Stability Pact" OR "NextGenerationEU" OR "budget deficit" OR '
    'Germany OR France OR Italy OR Spain OR election OR coalition OR "political crisis" OR "fiscal policy" OR '
    '"US elections" OR "Presidential debate" OR Congress OR "White House" OR '
    '"tax cuts" OR "stimulus package" OR "fiscal support" OR Trump OR "spending bill"'
    """ 

    # Keyword 5 - Geopolitics & Global Risk Sentiment
    """
    'Russia OR Ukraine OR China OR "Middle East" OR Israel OR Gaza OR Iran OR '
    '"trade war" OR tariffs OR sanctions OR embargo OR '
    '"risk sentiment" OR "risk-off" OR "risk-on" OR "safe haven" OR volatility OR '
    '"US-China tensions" OR Taiwan OR "South China Sea" OR '
    '"energy crisis" OR "gas supply" OR "oil prices" OR OPEC'
    """ 

    # Keyword 6 - Market & Yield Dynamics
    """
    '"bond yields" OR "yield spread" OR "2Y yields" OR "10Y yields" OR Treasuries OR Bunds OR Gilts OR '
    '"carry trade" OR "rate differential" OR "term premium" OR "yield curve inversion" OR "Yield Curve" OR'
    '"real yields" OR TIPS OR "curve steepening" OR "curve flattening" OR "UST 2Y" OR "UST 10Y" OR "TED spread"'
    """ 

    # Keyword 7 - Analysts, Forecasts & Market Commentary
    """
    '"FX strategist" OR "currency strategist" OR "market analyst" OR "forex forecast" OR '
    '"bank of america" OR "JP Morgan" OR "Goldman Sachs" OR "Deutsche Bank" OR "Citigroup" OR "UBS" OR "Morgan Stanley" OR '
    '"EUR/USD" OR forex OR currency OR '
    '"traders expect" OR "markets price in" OR positioning OR speculators OR '
    'CFTC OR "COT report" OR "commitment of traders"'
    """ 

    # Keyword 8 - Risk Metrics
    """
    '"VIX" OR "MOVE index" OR "FX volatility" OR "implied volatility" OR "realized volatility" OR '
    '"Treasury volatility" OR "risk appetite" OR "market sentiment" OR "safe haven demand" OR '
    '"credit spreads" OR "CDS spreads" OR "risk premium" OR '
    '"market liquidity" OR "funding stress" OR "dollar funding" OR "basis swaps" OR '
    '"S&P 500" OR "equity volatility"'
    """ 

    # Keyword 9 - Sentiment & Expectations
    """
    '"investor confidence" OR "consumer confidence" OR "economic sentiment" OR '
    '"ZEW index" OR "IFO index" OR Sentix OR "Michigan sentiment" OR '
    '"inflation expectations" OR "terminal rate" OR '
    '"forward guidance" OR "dot plot" OR "policy path"'
    """
