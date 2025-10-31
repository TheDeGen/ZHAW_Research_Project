# News Data Collection

This directory contains scripts and data for news articles for energy market analysis and details how we cleaned up the data

## Data Source
**NewsAPI.org** - Aggregates articles from major news outlets worldwide

## Date Coverage
- **Start Date:** November 1, 2020
- **End Date:** October 31st, 2025

# First Data Fetch and Cleanup
## Step 1) First English Sourced News
We used `english_news_fetcher.py` with the following parameters:
Keywords: Oil, Gas, Electricity, Energy, Market, Interest Rates, Tariffs
Sources: ABC News, Associated Press, BBC News, Bloomberg, CNN, NBC News, Reuters, The Wall Street Journal, The Washington Post, Time, USA Today

The result was us getting 57'169 News articles.

## Step 2: First Clean Up Round
In the `notebook data_analysis_v1.ipynb` we conducted preliminary data analysis. We also used BERTopic to check some of the topics in the dataset and found the following that do not fit:
1) `0_nfl_colts_free_draft` with keywords: **nfl**, **colts**, free, draft, bears, vs, **bengals**, **chiefs**
2) `3_music_film_movie_swift` with keywords: **music**, film, movie, swift, **taylor**, her, **album**, **hollywood**
3) `4_cup_league_manchester_england` with keywords: cup, league, manchester, england, **chelsea**, liverpool, **premier**, manager
4) `8_football_college_state_vs`with keywords: **football**, college, state, vs, channel, ten, **stream**, tv
5) `10_nba_lakers_celtics_nets`with keywords: **nba**, **lakers**, **celtics**, nets, **76ers**, bulls, boston, **sixers**

The words in **bold** are pretty obviously irrelevant to our project and used to remove headlines that we do not want in the dataset. Doing this helped us remove 4'020 articles, leaving us with 53'149. 

## Step 3: Second Clean Up Round
Still in the the `notebook data_analysis_v1.ipynb` we use BERTopic to pass on the dataset a second time to identify other large topics to remove. The following topics stood out to us as irrelevant:
1) `1_free_cowboys_bears_steelers`with keywords: free, **cowboys**, bears, **steelers**, agent, rams, draft, **saints**, **giants**, **eagles**
2) `4_cup_league_manchester_england`with keywords: **cup**, **league**, manchester, england, liverpool, **arsenal**, manager, win, **champions**, side
3) `5_film_movie_oscars_her`with keywords: **film**, **movie**, **oscars**, her, fashion, star, **netflix**, **festival**, **barbie**, **oscar**
4) `7_deals_best_shop_amazon`with keywords: deals, best, shop, amazon, **walmart**, day, save, sale, apple, **appliances**
5) `15_shooting_police_man_murder`with keywords: **shooting**, police, man, **murder**, station, suspect, old, **shot**, charged, woman

Using the same methodology as the step above, removing **bold** keywords from the dataset, we managed to remove 1'678 articles, leaving us with 51'471

# Step 4: Third Clean Up Round
Same methodology as above. Topics identified and to be removed:
1) `3_free_bears_rams_draft`with keywords: free, **bears**, **rams**, draft, agent, **packers**, **texans**, **lions**, **offseason**, **49ers**
2) `5_england_manchester_liverpool_league`with keywords: england, **manchester**, **liverpool**, **league**, manager, cup, club, win, side, **celtic**
3) `8_fashion_her_movie_pop`with keywords: **fashion**, her, **movie**, **pop**, **star**, **netflix**, **film**, show, box, **drama**
4) `12_best_deals_shop_amazon`with keywords: **best deals**, shop, **amazon**, day, **sale**, save, prime, apple, buy
5) `21_football_lsu_ten_usc`with keywords: **football**, lsu, ten, **usc**, **college**, portal, **penn**, state, oklahoma, transfer
6) `25_coffee_starbucks_drinks_beer` with keywords: **coffee**, **starbucks**, drinks, **beer**, drink, **alcohol**, **soda**, **caffeine**, **coca**, **cola**
7) `26_ufc_fight_mma_title` with keywords: **ufc**, fight, **mma**, title, bonus, vs, night, **champ**, **pereira**, **makhachev**
8) `28_telescope_nasa_astronomers_space` with keywords: **telescope**, **nasa**, **astronomers**, space, **webb**, solar, jupiter, lights, **eclipse**, scientists
9) `33_olympic_olympics_paris_marathon` with keywords: **olympic**, **olympics**, paris, **marathon**, gold, **medal**, **swimming**, team, **sport**, women
10) `35_skin_hair_best_your` with keywords: **skin**, **hair**, best, your, shop, **dermatologists**, dry, experts, brands, **sunscreen**
11) `36_police_tear_protesters_protests` with keywords: **police**, tear, protesters, protests, protest, **bangladesh**, clashes, gas, demonstrators, opposition
12) `38_nets_bulls_rockets_thunder` with keywords: **nets**, **bulls**, **rockets**, thunder, **lebron**, brooklyn, **warriors**, chicago, **nba**, james
13) `45_dog_watch_zoo_bear` with keywords: **dog**, watch, **zoo**, **bear**, **dogs**, **cat**, **retriever**, golden, **animals**, **eagle**
14) `49_retail_holiday_stores_sales` with keywords: retail, **holiday**, stores, sales, spending, retailers, **shoppers**, **shopping**, consumers, store

After this pass we managed to remove 7'092 articles, leaving us with a total 44'379 articles.