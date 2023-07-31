import pandas as pd
from bs4 import BeautifulSoup
import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from glob import glob


def get_team_stats(start_year=2015, end_year=2022, 
                   bat_or_pitch='batting', metrics=['standard'], full_seasons_only = True):
    """
        Reads and stores team batting or pitching stats from start_year to end_year.
        Can't merge pre-2015 data to 2015-present data because there are 2 new stats for post-2014 years
        Batting metrics: standard, advanced, sabermetric
        Pitching metrics: standard, advanced, batting (against), ratio
        Returns pandas DataFrame.
    """
    # for each year
    for year in range(start_year, end_year+1):
        # skip non-full-seasons
        if year in [2020, 2023]:
            continue   
        
        if bat_or_pitch == 'batting':
            # get Win-Loss records
            url = f'https://www.baseball-reference.com/leagues/majors/{year}-standard-pitching.shtml'
            w_l_records = read_website_tables(url, webdriver_path = '~/chromedriver')[0][:-3][['W','L','W-L%']] 

        # get stats by metric
        metric_dfs =[]
        for metric in metrics:
            # read and store team stats table for given year. 2 tables are on this url with the first being team stats
            url = f'https://www.baseball-reference.com/leagues/majors/{year}-{metric}-{bat_or_pitch}.shtml'
            metric_df = read_website_tables(url, webdriver_path = '~/chromedriver')[0][:-3]

            if metric == 'advanced':
                metric_df.columns = [col[1] for col in metric_df.columns]
            
            metric_dfs.append(metric_df)
        
        # join tables to get stats for the year
        year_stats = pd.concat(metric_dfs, axis=1)
        
        # add year column
        year_stats['year'] = year
#         # move column to be first
#         cols = list(year_stats.columns)
#         cols.insert(0, cols.pop(cols.index('year')))
#         year_stats = year_stats[cols]
        
        # if team_stats df exists, add year_stats to team_stats
        if 'team_stats' in locals():
            team_stats = pd.concat([team_stats, year_stats])
        # else initialize it
        else:
            team_stats = year_stats
        
    return team_stats


def get_player_stats(start_year=2015, end_year=2022, 
                   bat_or_pitch='batting', metric='standard', full_seasons_only = True):
    """
        Reads and stores player batting or pitching stats from start_year to end_year.
        Can't merge pre-2015 data to 2015-present data because there are 2 new stats for post-2014 years
        Batting measures: standard, advanced, sabermetric
        Pitching measures: standard, advanced, batting (against), ratio
        Returns pandas DataFrame.
    """ 
        
    # for each year
    for year in range(start_year, end_year+1): 
        # skip non-full-seasons
        if year in [2020, 2023]:
            continue

        url = f'https://www.baseball-reference.com/leagues/majors/{year}-{metric}-{bat_or_pitch}.shtml'

        # read and store player stats table for given year. 2 tables are on this url with the second being player stats
        year_stats = read_website_tables(url, webdriver_path = '~/chromedriver')[1][:-1]

        # if metric is advanced, rename columns to be second element of column names
        if metric == 'advanced':
            year_stats.columns = [col[1] for col in year_stats.columns]

        # remove non-player rows
        year_stats = year_stats[~(year_stats['Rk'] == 'Rk')]

        # add year column
        year_stats['year'] = year

        # add player ID column, must reset index
        year_stats = year_stats.reset_index()
        year_stats['id'] = pd.DataFrame(read_player_ids(url), columns=['id'])

        # if player_stats df exists, add year_stats to player_stats
        if 'player_stats' in locals():
            player_stats = pd.concat([player_stats, year_stats])
        # else initialize it
        else:
            player_stats = year_stats
        
    return player_stats


def read_player_ids(url, webdriver_path = '~/chromedriver'):
    """
        reads player ID's from a url and returns these ID's in a list of strings
    """
    # Read url and return parsed HTML
    soup = read_website(url)
    
    # Find all the table elements
    table_elements = soup.find_all('table')

    # Second table in url has player_stats
    player_table = table_elements[1]
    
    # player IDs are in 'data-append-csv' attribute of 'td' tags that have 'data-stat = player'
    # emit last ID, which is blank
    player_ids = [tag.get('data-append-csv')
                 for tag in player_table.find_all('td', attrs={'data-stat': 'player'})][:-1]
    
    return player_ids


def read_website_tables(url, webdriver_path = '~/chromedriver'):
    """
        reads table elements from a url and returns these tables in a list of DataFrames
    """
    # Read url and return parsed HTML
    soup = read_website(url)

    # Find all the table elements
    table_elements = soup.find_all('table')

    # Extract the HTML content from each table element
    html_tables = [str(table) for table in table_elements]

    # Pass the list of HTML tables to pd.read_html()
    dfs = pd.read_html('\n'.join(html_tables))
    
    return dfs


def read_website(url, webdriver_path = '~/chromedriver'):
    '''
        read website and returns a data structure representing a parsed HTML document
    '''
    # Path to chromedriver executable
    webdriver_path = webdriver_path

    # Set up the Selenium driver with options
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    driver = webdriver.Chrome(service=Service(webdriver_path), options=options)
    
    # Load the webpage
    driver.get(url)

    # Wait for the dynamic content to load (if necessary)
    # You can use driver.implicitly_wait() or other wait methods

    # Extract the page source after the dynamic content has loaded
    source = driver.page_source

    # Close the Selenium driver
    driver.quit()
    
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(source, 'lxml')

    return soup

def save_data(df, filename):
    """
        store DataFrame to csv file
    """
    df.to_csv(f'{filename}.csv', index=False)

def get_mlb_acronyms():
    """
        reads 30 team acronyms from url and stores them into a list
        Returns list
    """
    url = 'https://www.baseball-reference.com'
    source = requests.get(url).text # html of website
    
    soup = BeautifulSoup(source, 'lxml')
    
    # All team acronyms are in the first `div` tag with  
    # class = "formfield". They are then in option tags (except the first).
    div_formfield = soup.find_all('div', class_='formfield')[0]
    option_tags = div_formfield.find_all('option')[1:]
    
    # Should be 30 acronyms
    team_acronyms = [str(tag)[15:18] for tag in option_tags]

    return team_acronyms


def merge_dfs(csv_files=glob("data/*.csv")):

    # Read csv's to DataFrames. 
    # Store in dictionary where key is filename and value is the DataFrame
    csv_dfs = {}
    for file in csv_files:
        # Create filename string which will be the name of the key
        filename = os.path.basename(file).replace(".csv", "")
        # Read file into DataFrame
        file_df = pd.read_csv(file)
        # Store DataFrame as a value
        csv_dfs[filename] = file_df

    # merge team stats
    # team stats
    # list 'team_batting' df's, concat them, and sort by year
    team_batting_stats = pd.concat([df for k, df in csv_dfs.items() if k.startswith('team_batting')]).sort_values('year')
    # list 'team_pitching' df's, concat them, and sort by year
    team_pitching_stats = pd.concat([df for k, df in csv_dfs.items() if k.startswith('team_pitching')]).sort_values('year')
        
    # merge player stats
    # player stats
    player_batting_standard = pd.concat([df for k, df in csv_dfs.items()
                                           if k.startswith('player_batting_standard')]).sort_values('year')
    player_batting_advanced = pd.concat([df for k, df in csv_dfs.items()
                                           if k.startswith('player_batting_advanced')]).sort_values('year')
    player_batting_sabermetric = pd.concat([df for k, df in csv_dfs.items()
                                           if k.startswith('player_batting_saber')]).sort_values('year')
    
    player_pitching_standard = pd.concat([df for k, df in csv_dfs.items()
                                           if k.startswith('player_pitching_standard')]).sort_values('year')
    player_pitching_advanced = pd.concat([df for k, df in csv_dfs.items()
                                           if k.startswith('player_pitching_advanced')]).sort_values('year')
    
    team_batting_stats.to_csv('data/team_batting_stats.csv', index=False)
    team_pitching_stats.to_csv('data/team_pitching_stats.csv', index=False)
    player_batting_standard.to_csv('data/player_batting_standard.csv', index=False)
    player_batting_advanced.to_csv('data/player_batting_advanced.csv', index=False)
    player_batting_sabermetric.to_csv('data/player_batting_sabermetric.csv', index=False)
    player_pitching_standard.to_csv('data/player_pitching_standard.csv', index=False)
    player_pitching_advanced.to_csv('data/player_pitching_advanced.csv', index=False)