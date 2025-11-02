# News Data Collection

This directory contains scripts and data for news articles for energy market analysis and details how we cleaned up the data

## Data Source
**NewsAPI.org** - Aggregates articles from major news outlets worldwide

## Date Coverage
- **Start Date:** November 1, 2020
- **End Date:** October 31st, 2025

# First English News Data Bacth Fetch and Cleanup
## Step 1) First English Sourced News
We used `english_news_fetcher.py` with the following parameters:
Keywords: Oil, Gas, Electricity, Energy, Market, Interest Rates, Tariffs
Sources: ABC News, Associated Press, BBC News, Bloomberg, CNN, NBC News, Reuters, The Wall Street Journal, The Washington Post, Time, USA Today

The result was us getting 57'169 News articles. Results are saved in `english_news_raw.csv`

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

# Second English News Data Bacth Fetch and Cleanup
## Step 1: Fetching News
Same as in the first batch we use `english_news_fetcher.py` with the following parameters:
Keywords:Renewable, solar, wind, hydro, nuclear, coal, carbon, emissions, climate, policy,supply, OPEC, IEA, ECB, European Comission, EPEX, European Union, ACER, pipeline, grid, transmission, infrastructure, storage.
Sources: Same as in the first batch

This resulted in us getting 61'346 articles and saved in `english_news2_raw.csv`

## Step 2: Deduplication & Merging
All of the following steps will be done in ?data_analysis_v2.ipynb`

We merge this new dataset with the filtered last batch. We are aware that some articles that were already filtered might get back into the dataset. But we'll feilter it again using BERTopic.

After merging and deduplication we get `english_news_v4` with 89'517 articles. 16'208 duplicates were removed.

## Step 3: First Cleanup Round
Similar to the previous batch, we use BERTopic to analyse the contents of the new merged dataset. The following topics were found to be irrelevant:
1) `0_league_cup_england_champions` with keywords: **league**, **cup**, england, **champions**, **wales**, **rugby**, **manchester**, final, euro, manager
2) `4_swift_taylor_film_movie` with keywords: **swift**, **taylor**, **film**, **movie**, her, **oscars**, **star**, show, **music**, **netflix**
3) `5_transgender_students_school_education` with keywords: **transgender**, **students**, school, education, **trans**, **gender**, university, schools, harvard, **dei**
4) `7_immigration_border_ice_deportation` with keywords: **immigration**, border, ice, **deportation**, **deportations**, **migrants**, **immigrants**, guard, administration, trump
5) `11_harris_kamala_vice_democratic` with keywords: harris, kamala, vice, democratic, **campaign**, her, **dnc**, she, presidential, president
6) `13_nfl_free_draft_patriots` with keywords: **nfl**, free, **draft**, patriots, **jaguars**, **broncos**, agent, **raiders**, **panthers**, **vikings**
7) `14_shooting_police_man_death` with keywords: **shooting**, police, man, death, **murder**, officer, suspect, charged, officers, **killing**
8) `16_biden_joe_debate_address` with keywords: biden, joe, debate, address, president, his, union, campaign, state, **reelection**
9) `18_deals_best_shop_amazon` wtih keywords: deals, best, shop, **amazon**, save, sale, **cyber**, apple, day, buy
10) `24_championships_gold_gb_britain` with keywords: **championships**, gold, gb, britain, great, wins, european, **medal**, silver, bronze

Using a similar approach to the first batch, we manage to remove 7'143 articles, leaving us with 82'374. Results saved in `english_news_v5`

## Step 4: Second Cleanup Round
Implementing the same methodology as before we identify the following irrelevant topics:
1) `0_england_league_euro_cup` with keywords: england, **league**, euro, **cup**, manager, **manchester**, **champions**, **club**, ireland, boss
2) `5_strike_workers_union_uaw` with keywords: strike, workers, union, **uaw**, contract, pay, labor, auto, tentative, strikes
3) `9_harris_kamala_vice_democratic` with keywords: **harris**, **kamala**, vice, democratic, **dnc**, her, campaign, she, presidential, president
4) `10_show_her_movie_gma` with keywords: show, her, **movie**, gma, box, **oscar**, watch, **fashion**, gala, **beyoncé**
5) `11_dies_died_carter_jimmy` wtih keywords: dies, died, **carter**, **jimmy**, age, former, aged, who, **hackman**, at
6) `14_google_twitter_meta_antitrust` with keywords: google, twitter, meta, **antitrust**, facebook, musk, apple, elon, **social**, media
7) `16_airlines_flights_boeing_flight` with keywords: **airlines**, flights, **boeing**, flight, airport, **airline**, air, airbus, passengers, attendants
8) `17_deals_best_shop_amazon` with keywords: **deals**, best, shop, amazon, save, sale, **cyber**, day, apple, buy
9) `18_patriots_free_seahawks_dolphins` with keywords: **patriots**, free, **seahawks, dolphins, browns, roster**, agent, ravens, bills, 49ers
10) `19_biden_joe_debate_address` with keywords: biden, joe, debate, address, president, his, union, he, **campaign**, state
11) `23_bankruptcy_mcdonald_stores_prices` with keywords bankruptcy, **mcdonald**, stores, prices, food, grocery, chain, **lobster**, chapter, files
12) `26_abortion_pill_abortions_access` with keywords: **abortion**, pill, **abortions**, access, **mifepristone**, roe, **ivf**, supreme, court, ban
13) `28_ice_guard_khalil_immigration` with keywords: ice, guard, **khalil**, **immigration**, **mahmoud**, chicago, enforcement, agents, student, national
14) `29_prix_grand_verstappen_formula` with keywords: **prix**, grand, **verstappen**, formula, max, bull, **f1**, **hamilton**, **ferrari**, red
15) `30_education_school_schools_harvard` with keywords: **education**, school, schools, **harvard**, **university**, department, college, admissions, supreme, court
16) `32_niger_coup_african_gabon` with keywords: **niger**, coup, african, **gabon**, junta, africa, election, military, president, presidential
17) `34_california_court_montana_supreme` with keywords: california, court, montana, supreme, pipeline, gas, environmental, climate, state, lawsuit
18) `44_championships_gold_gb_britain` with keywords: **championships**, gold, gb, britain, wins, **medal**, silver, european, great, win
19) `46_dog_watch_retriever_dogs` with keywords: **dog**, watch, **retriever, dogs, puppy**, golden, **cat**, owner, **adorable**, her
20) `47_nasa_telescope_asteroid_mars` with keywords: **nasa, telescope, asteroid, mars**, earth, planet, **space, astronomers, webb, spacecraft**

Using the same approach to the first batch, we manage to remove 6'443 articles, leaving us with 75'931. Results saved in `english_news_v6`

## Step 5: Third Cleanup Round
We realised that during keywor removal, we only looked at the titles, not combined text (Title + Description), and running BERTopic again we saw a lot of topic appear again. Which means we will run the removal function with the same keywords as in step 4, but will apply it to the description of the news data (since titles have been removed already).

This allowed us to remove 7'015 articles, leaving us with 68'916 remaining. Results are saved in `english_news_v7`

## Step 6: Fourth Cleanup Round
Same methdology as above. We identified the following topics to remove, This time with a more detailed reasoning for certain topics that might seem relevant:
1) `2_england_euro_manager_ireland` Keywords to remove: rugby, premiership, football
2) `3_mortgage_housing_rates_home` Keywords to remove: mortgage, housing, homebuyer, estate. These news are mainly about the US Housing market, which don't have a major impact on German Energy prices.
3) `4_snow_tornadoes_storm_storms`Keywords to remove: tornadoes, snow, storms, midwest. These weather news are mostly US-Based. Any weather events affecting Germany should be picked up when going through German news data fetching
4) `6_korea_north_korean_kim`Keywords to remove: korea, korean, missile
5) `7_heat_temperatures_hottest_record`Keywords to remove: heat, temparatures, summer, temparature, heatwave. Same reasoning as in 3
6) `8_strike_workers_union_pay` Keywords to remove: strike, workers, nurses, doctors
7) `9_classified_case_trump_court` Keywords to remove: classified
8) `13_biden_joe_debate_address` Keywords to remove: debate, union
9) `19_bill_tax_senate_republicans` Keywords to remove: senate, beautiful, megabill. This topic is mostly about the BBB, with those keywords we should be able to filter them out
10) `21_her_season_show_kendrick` Keywords to remove: Kendrick, concert, tv, lotus, lamar, songs
11) `23_water_river_colorado_drought` Keywords to remove: colorado, drought, lake, arizona. Same reason as 3
12) `24_free_ravens_bills_agent` Keywords to remove: ravens, titans, 49ers, chargers
13) `28_wildfire_wildfires_fire_california` Keywords to remove: wildfire, wildfires, firefighters, angeles. Same reason as 3
14) `29_football_college_ten_sec` Keywords to remove: football, college, ohio, coach
15) `30_dies_died_aged_age` Keywords to remove: kissinger, schultz
16) `32_crash_plane_helicopter_crashed` Keywords to remove: helicopter, pilot
17) `42_ipo_adani_india_billion` Keywords to remove: ipo, adani, ipos
18) `43_watch_her_dad_dance` Keywords to remove: dance, dad, mom, baby
19) `44_jr_rfk_kennedy_robert` Keywords to remove: rfk
20) `48_haley_nikki_primary_republican` Keywords to remove: haley, nikki

This time also checking in the description, we managed to remove 12'854 articles, leaving 56'062 in the dataset. Results are saved in `english_news_v8`

## Step 6: Fifth and Final Cleanup Round
Same methodology, we identified the following topics to remove:
1) `2_vs_players_game_bowl` remove: game, kraken, nhl, season, basketball
2) `3_england_euro_manager_ireland` remove: manager, chelsea, scotland, liverpool
3) `4_hunter_court_trial_trump` remove: hunter, impeachment
4) `13_california_court_supreme_lawsuit` remove: lawsuit, montana
5) `16_meat_diet_you_eating` remove: meat, diet, beef, healthy
6) `17_stores_grocery_costco_store` remove: grocery, costco, retailer
7) `19_home_homes_rent_prices` remove: homes, renters, mortgage
8) `22_drug_weight_drugs_loss` remove: weight, ozempic, diabetes, alzheimer
9) `27_season_her_gma_love` remove: gma, season, actor, star, film
10) `33_museum_auction_art_painting` remove: museum, art, heist
11) `34_disney_disneyland_walt_resort` remove: disney, disneyland, walt, foodie
12) `39_dies_died_age_former` remove: shultz, 88
13) `41_hurricane_tropical_storm_atlantic` remove: hurricane, storm
14) `42_king_charles_iii_coronation` remove: king, charles, prince
15) `43_podcast_bloomberg_aspect_understand` remove: podcast, sweeny, radio
16) `47_ufc_fight_252_seattle` remove: ufc, 252
17) `48_jr_kennedy_rfk_robert` remove: kennedy, rfk, hhs, maha

This allowed us to remove 7'735 articles, leaving us with 48'327 in the dataset. Results saved as `english_news_v9`

## Step 7: Sanity Check
We ran DeBERTa for zero-shot classification to quickly check our news ditribution in a more robust manner. Results are as follows:
- geopolitical news: 25'307
- financial markets: 8'574
- other: 5'657
- entertainment: 3'408
- sports: 2'071
- weather: 1'532
- electricity or energy production: 1'421
- electricity or energy consumption: 356

This shows that about 23% of the data in `english_news_v9` is still a little irrelevant. We'll deal with this later though.

## Step 8: Sixth Cleanup Round
Same methodology, we identified the following topics to remove:
1) `2_zoo_watch_species_bear` remove: zoo, bear, whale, shark
2) `18_drug_drugs_weight_loss` remove: drugs, ozempic, diabetes, alzheimer
3) `20_gun_school_lgbtq_marriage` remove: lgbtq, marriage
4) `29_museum_auction_art_painting` remove: museum, art, painting
5) `47_ufc_fight_seattle_252`remove: ufc

This allowed us to remove 6'504 articles, leaving us with 41'823 in the dataset. Results saved as `english_news_v10`

## Step 9: Seventh Cleanup Round
Same methodology, we identified the following topics to remove:
1) `1_disney_her_watch_you` remove: disney, disneyland
2) `9_vs_players_free_jets`remove: espn
3) `20_venezuela_maduro_venezuelan_opposi...` remove: maduro
4) `34_man_police_officers_death` remove: police
5) `39_sudan_congo_sudanese_darfur` remove: sudan, congo, ethiopia


This allowed us to remove 1'048 articles, leaving us with 40'775 in the dataset. Results saved as `english_news_v11`

---
# First German News Data Bacth Fetch and Cleanup
## Step 1: Initial Data fetch
Similar setup to pulling a broad amount of German news for 5 years. We do it in 2 steps, then combining both datasets. Sources remain the same for both steps.

Sources: `'bild, der-tagesspiegel, die-zeit, focus, handelsblatt, spiegel-online, wirtschafts-woche'`

Keywords #1:
```
'Strom OR Energie OR Kraftwerk OR Erneuerbare OR Atomkraft OR Kohlekraftwerk OR Gaskraftwerk OR Windkraft OR '
'Solar OR Energiepolitik OR Strommarkt OR Energienetz OR Stromnetz OR Strompreis OR Energiepreis OR Stromverbrauch OR '
'Energieverbrauch OR EZB OR Stromrechnung OR Energiekosten OR Energiekrise OR Stromausfall OR Energieversorgung OR '
'Heizung OR Klimaanlage OR Kältewelle OR Hitzewelle OR Markt OR Preis OR Politik OR Regulierung OR Stromversorgung'
```
This returned 74'807 news articles. Results are saved as `german_news_raw`

Keywords #2:
```
'Stromerzeugung OR Stromnachfrage OR Strombedarf OR Stromsparen OR Stromeffizienz OR Stromsparen OR Stromsparen OR Stromsparen OR'
'Wetter OR EZB OR EPEX OR Zentralbank OR Erdöl OR Erdgas OR Kohle OR OPEC OR IEA OR EU'
```
This returned 58'637 News. Results are saved as `german_news_raw2`

## Step 2: Merging & De-Duplication
Similar to what we did with the english news, we merge both datasets and remove any duplicates. We match on `url` for exact matches. All of this and next steps will be done in `data_analysis_v3.ipynb`.

Post merging we have 133'444 articles, with 11'236 duplicates removed, meaning 122'208 articles remain. This dataset will be saved as `german_news_raw_marged`

## Step 3: Initial Cleanup round
We're doing a similar process to what we did with the english dataset, by using BERTopic to finad and classify the top 50 topics. Since the headlines are in German we will be using `paraphrase-multilingual-MiniLM-L12-v2`. We found the following irrelevant topics:
1) `3_kriminalität_polizei_mann_prozess` remove: täter, kokain, drogen, messer
2) `9_schulen_eltern_kinder_schule` remove: schulen, eltern, bildung, lehrer, mutter
3) `16_formel_verstappen_hamilton_rennen` remove: formel, verstappen, hammlton, schumacher
4) `17_tiere_wolf_wölfe_wölfen` remove: tiere, wolf, wölfe, hund, tierschutz, schweinpeste, hunde
5) `29_weihnachtsmarkt_weihnachten_weihna` remove: weihnachtsmarkt, weihnachten, glühwein
6) `33_tourismus_urlaub_reisen_touristen` remove: tourismus, urlaub, hotels
7) `42_mali_sudan_niger_bundeswehr` remove: mali, sudan, niger, kongo

We're keeping sport for now since one paper did find a link to energy usage. After running this first cleanup script we removed 5'070 articles, with 117'138 remaining. Results are saved in `german_news_v1.csv`

## Step 4: Second Cleanup round
Similar to the methodology above, we run BERT to identify topics.