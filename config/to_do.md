
## Assess scrapers
The goal is to assess which scrapers are the most reliant and can pull the most data using [soccerdata]
(https://github.com/probberechts/soccerdata) and the documentation https://soccerdata.readthedocs.io/en/latest/

Last season I used fbref but there are two issues. The player names in fbref are sometimes different than the player names in sleeper (see next point) and there has been some issues with recent rissilience with cloudflare making it difficult to scrape. However based on this, we should still test which is best. 
The main goal is to ensure that we can pull current seasons team and player statistics and also if possible statistics from previous years

## Assess naming of players for different scrapers
The naming of players can sometimes divert and we should assess how we get the naming of players from other sources as correctly as the player names in Sleeper using e.g.
def get_sleeper_players(league: str = "clubsoccer:epl"):
    """Fetch EPL players from Sleeper API and return as DataFrame."""
    url = f"https://api.sleeper.app/v1/players/{league}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    players = response.json()
    return pd.DataFrame(players).T

However, we should not spam this endpoint and therefore cache it so we at most look up players once a week.
Based on this result we should decide which scrapper(s) are most suitable.

## Rework statistics
The goal is to better assess how my roster should look like based on more thorugh statistics and data scraping.
The idea is to have as much data as possible for a given player to maximize the amount of points I will get in a game week using the sleeper app. Here we should try to convert different data into an expected points based on the scoring_settings in
league_id = 1385257445575639040
league = requests.get(
    f"https://api.sleeper.app/v1/league/{league_id}"
).json()

The idea is to use the best scrapers from [soccerdata](https://github.com/probberechts/soccerdata). These should have been assessed in (Assess scrapers)

We should also assess if it it is possible to get event data, tracking data, etc. using e.g.
https://github.com/PySport/kloppy and https://kloppy.pysport.org/user-guide/getting-started/

Based on the assessment we should finalize the statistics.


## Snake draft
We are 10 players in a random snake draft format and I will be assigned a random number for when it is my turn to pick a player.
The lesson from last time is that I need to have several options of players during the snake draft as the script will sometimes take time to run. Therefore I would need to have e.g. top10 picks so if an oponent chooses the player I intended to pick, then I have backup players to choose from.


## Game week roster
Each new game week, I will be running the script to assess
- evaluate if players from the bench should be added in 
- evaluate if we need to trade players
- if player trade, check the taken players by my opponents 

## Last call change for maximum points in a game week
- What players are going to play in a match (confirmation 1 h before)
- How does that reflect my roster, i.e which players should I pick from my current roster or does it make sense to pick new players for the game week
- Last call change against machup ("https://api.sleeper.app/v1/league/<league_id>/matchups/<week>")
- Given my oponents roster picks, how should I pick the formation and my players?


## Flask/Streamlit application 
- make a flask/streamlit application for the change in rosters over the weeks and for my oponents
- the flask/streamlit application should also have statistics for each game week and the different player performances
- the flask applicationstreamlit should also have a sum of the statistics per team and per player

