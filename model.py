"""
CSGO Match Predictor
-Matt Stroud
"""

# Imports
# import numpy as np
from datetime import datetime as dt
from dateutil.parser import parse
import dateutil.relativedelta
import os
import urllib3
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import tqdm
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# Request Handling
def request(url):
    time.sleep(1)
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    html = r.data.decode("utf8")

    return html


# HLTV Data Scraping
#
# Sample: https://www.hltv.org/stats/teams/ftu?startDate=2022-11-25&endDate=2023-01-25&maps=de_inferno&rankingFilter=Top5
#
# Base URL
HLTV_BASE = "https://www.hltv.org/stats/teams"
# Stat Categories
HLTV_STAT_OPTIONS = ["", "/ftu", "/pistols"]
# Ranking Filter
HLTV_RANKING_FILTERS = {
    'all': "all",
    '5': "Top5",
    '10': "Top10",
    '20': "Top20",
    '30': "Top30",
    '50': "Top50"
}
# Maps
HLTV_MAPS = {
    'anc': "de_ancient",
    'anb': "de_anubis",
    'inf': "de_inferno",
    'mrg': "de_mirage",
    'nke': "de_nuke",
    'ovp': "de_overpass",
    'vtg': "de_vertigo"
}
# Sides
HLTV_SIDES = ["COUNTER_TERRORIST", "TERRORIST"]
# Teams
HLTV_TEAMS = {
    'faze': "FaZe",
    'navi': "Natus Vincere",
    'g2': "G2",
    'ence': "ENCE",
    'nip': "Ninjas in Pyjamas",
    'mouz': "Mouz",
    'liquid': "Liquid",
    'og': "OG",
    'astralis': "Astralis",
    'big': "BIG",
    'outsiders': "Outsiders",
    'vitality': "Vitality",
    'fnatic': "fnatic",
    'furia': "FURIA",
    'heroic': "Heroic",
    'bne': "Bad News Eagles"
}
HLTV_DATA_FEATURES = [
    "teama_maps",
    "teama_kddiff",
    "teama_kdr",
    "teama_rtg",
    "teama_ct_rw",
    "teama_ct_opk",
    "teama_ct_multik",
    "teama_ct_5v4",
    "teama_ct_4v5",
    "teama_ct_trd",
    "teama_ct_adr",
    "teama_ct_fa",
    "teama_t_rw",
    "teama_t_opk",
    "teama_t_multik",
    "teama_t_5v4",
    "teama_t_4v5",
    "teama_t_trd",
    "teama_t_adr",
    "teama_t_fa",
    "teama_ct_pistol_rw",
    "teama_ct_pistol_r2conv",
    "teama_ct_pistol_r2brk",
    "teama_t_pistol_rw",
    "teama_t_pistol_r2conv",
    "teama_t_pistol_r2brk",
    "teamb_maps",
    "teamb_kddiff",
    "teamb_kdr",
    "teamb_rtg",
    "teamb_ct_rw",
    "teamb_ct_opk",
    "teamb_ct_multik",
    "teamb_ct_5v4",
    "teamb_ct_4v5",
    "teamb_ct_trd",
    "teamb_ct_adr",
    "teamb_ct_fa",
    "teamb_t_rw",
    "teamb_t_opk",
    "teamb_t_multik",
    "teamb_t_5v4",
    "teamb_t_4v5",
    "teamb_t_trd",
    "teamb_t_adr",
    "teamb_t_fa",
    "teamb_ct_pistol_rw",
    "teamb_ct_pistol_r2conv",
    "teamb_ct_pistol_r2brk",
    "teamb_t_pistol_rw",
    "teamb_t_pistol_r2conv",
    "teamb_t_pistol_r2brk",
    "match_winner"
]


# Scrape Data From HLTV


def scrapeData(team, html):

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text().strip()
    arr = []
    for i in re.split('\n\n', text):
        try:
            if i.split('\n')[1]==team:
                arr.append(i)
        except: pass
    data = [i for i in arr[0].split('\n') if i and i != team]

    return data


def hltvTeamData(team, start=False, end=False, rank='all', maps=[]):
    # Build URLs
    if start and end:
        dateFilter = f"startDate={start}&endDate={end}"
    else:
        dateFilter = "startDate=all"
    rankFilter = f"rankingFilter={HLTV_RANKING_FILTERS[rank]}"
    mapFilter = ""
    if maps:
        for map in maps:
            mapFilter.append(f"&maps={HLTV_MAPS[map]}")
    urls = []
    for category in HLTV_STAT_OPTIONS:
        if category:
            for side in HLTV_SIDES:
                urls.append(
                    f"{HLTV_BASE}{category}?{dateFilter}&{rankFilter}{mapFilter}&side={side}")
        else:
            urls.append(
                f"{HLTV_BASE}{category}?{dateFilter}&{rankFilter}{mapFilter}")

    # Scrape Data
    teamData = []
    for url in urls:
        teamData.append(scrapeData(team, request(url)))

    # Organize Data for Analysis
    cleanData = []
    for i in teamData:
        for j in i:
            if '%' in j:
                cleanData.append(float(j.strip('%')) / 100)
            elif re.match(".+\-.+", j):
                continue
            else:
                cleanData.append(float(j))

    # Remove duplicate map counts
    rmv = [4, 13, 22, 26]
    for i in sorted(rmv, reverse=True): del cleanData[i]

    # Return
    return cleanData


# Create Training Data
#
# Find list of matches played by relevant teams and grab input data based on the date as well as the result.
#
def matchesFromEvent(event):

    soup = BeautifulSoup(request(event), "html.parser")
    matches = [f"https://www.hltv.org{i['href']}" for i in soup.find("div", {"class": "results-all"}).find_all("a")]

    return matches


def matchTrainingData(match):

    # Request
    html = request(match)
    soup = BeautifulSoup(html, "html.parser")

    # Get teams and date from match page
    matchInfo = soup.find("div", {"class": "teamsBox"})
    date = parse(matchInfo.find("div", {"class": "date"}).get_text()).date()
    teams = [team.get_text() for team in matchInfo.find_all("div", {"class": "teamName"})]
    result = 0 if matchInfo.find("div", {"class": "team1-gradient"}).find("div", {"class": "won"}) else 1

    # Get start and end date for query from date
    end = str(date)
    start = str(date - dateutil.relativedelta.relativedelta(months=6))

    # Get data for each team and package for training purposes
    data = [np.array(hltvTeamData(team, start, end, '30')) for team in teams]
    data = np.append(np.array(data).flatten(), result)

    datadf = pd.DataFrame(data.reshape(1,-1), columns=HLTV_DATA_FEATURES)

    return datadf


def generateTrainingData():
    
    # Get Baseline data
    events = [
        "https://www.hltv.org/results?event=6970",
        "https://www.hltv.org/results?event=6349",
        "https://www.hltv.org/results?event=6348",
        "https://www.hltv.org/results?event=6586",
        "https://www.hltv.org/results?event=6141",
        "https://www.hltv.org/results?event=6346",
        "https://www.hltv.org/results?event=6140",
        "https://www.hltv.org/results?event=6345",
        "https://www.hltv.org/results?event=6372",
        "https://www.hltv.org/results?event=6137",
        "https://www.hltv.org/results?event=6136"
    ]

    results = []
    for event in events:
        for result in matchesFromEvent(event):
            results.append(result)
        
    data = []
    for result in tqdm.tqdm(results, desc="Generating training dataset"):
        try:
            data.append(matchTrainingData(result))
        except: pass

    datadf = pd.concat(data, ignore_index=True)
    # Prepare data for model
        # Normalize
        # Train/Test splits

    return datadf


# Model Creation
#
#
def generateModel(inp):

    model = keras.Sequential([
        keras.Input(shape=(inp,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Training Routine
#
#
def train(model, data):

    model.fit(data)

    return


# Predictor
#
#
def predictResult(model, match):

    matchData = matchTrainingData(match) # TODO Single function to get result when there is one and pop the column when there isn't

    prediction = model.predict(matchTrainingData)

    return prediction



# Main Function
if __name__ == "__main__":

    # test = "https://www.hltv.org/stats/teams/ftu?startDate=2022-10-26&endDate=2023-01-26"
    # testReq = request(test)
    # stew = scrapeData("BIG", testReq)
    # print(stew)

    # faze = hltvTeamData(HLTV_TEAMS['faze'], "2022-10-26", "2023-01-26", "20")
    # og = hltvTeamData(HLTV_TEAMS['og'], "2022-10-26", "2023-01-26", "20")
    # print(faze)

    # match = "https://www.hltv.org/matches/2361065/big-vs-liquid-blast-premier-spring-groups-2023"
    # match = "https://www.hltv.org/matches/2361064/complexity-vs-evil-geniuses-blast-premier-spring-groups-2023"
    # trainingData = matchTrainingData(match)
    # print(trainingData)

    # data = generateTrainingData()
    # print(data.shape)
    # data.to_csv('training.csv', index=False)

    data = pd.read_csv('training.csv')
    trainX, testX = train_test_split(data, test_size=0.2)
    trainY = trainX.pop('match_winner')
    testY = testX.pop('match_winner')

    model = generateModel(trainX.shape[1])
    print(model.summary())
    model.fit(trainX, trainY, epochs=200, batch_size=1)
    testLoss, testAcc = model.evaluate(testX, testY, verbose=1)
    print('Accuracy: ', testAcc)