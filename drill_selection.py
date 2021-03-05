import pickle
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
# Defining percentage functions 
def putts_inside_of_ten(PuttsInsideTenFeetAttempted, PuttsInsideTenFeetMade):
    ten_feet_percentage = (100 * (PuttsInsideTenFeetMade / PuttsInsideTenFeetAttempted))
    return ten_feet_percentage

def up_and_down(UpAndDownAttempts, UpAndDownSuccess):
    up_and_down_percentage = (100 * (UpAndDownSuccess / UpAndDownAttempts))
    return up_and_down_percentage

# Input of round
def round_entry(Putts, PuttsInsideTenFeetAttempted, PuttsInsideTenFeetMade, Fairways, Greens, Inside100, UpAndDownAttempts, UpAndDownSuccess):
    PuttsInsideTenFeetMakePercentage = putts_inside_of_ten(PuttsInsideTenFeetAttempted, PuttsInsideTenFeetMade)
    UpAndDownSuccessPercentage = up_and_down(UpAndDownAttempts, UpAndDownSuccess)
    return np.array([Putts, PuttsInsideTenFeetAttempted, PuttsInsideTenFeetMade, PuttsInsideTenFeetMakePercentage, Fairways, 
    Greens, Inside100, UpAndDownAttempts, UpAndDownSuccess, UpAndDownSuccessPercentage]).reshape(1,-1)


def predict(round_x):
    with open('model.golf', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    prediction = model.predict(round_x)
    prediction = pd.DataFrame(prediction)
    prediction = prediction.rename(columns = {0: 'Above or Below'})
    prediction['Above or Below'] = prediction['Above or Below'].apply(lambda x: 'Above' if x == 1 else 'Below')
    percent_chances = model.predict_proba(round_x)
    percent_chances = pd.DataFrame(percent_chances)
    percent_chances = percent_chances.rename(columns = {0 :'% chance of Below Average', 1: '% chance of Above Average'})
    game_with_percent_chances = [prediction, percent_chances]
    final_prediction = pd.concat(game_with_percent_chances, axis = 1)
    return final_prediction

def importance(round_x):
    with open('model.golf', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    golf_scores = pd.read_csv('jake_golf_scores.csv')
    golf_scores_num = golf_scores.drop(columns = 'DateOfRound')
    # Create new data frame for comparison to average
    golf_scores_with_average = golf_scores.copy()
    golf_scores_with_average['AverageScore'] = golf_scores_with_average.Score.mean()
    golf_scores_with_average['AboveOrBelowAverage'] = golf_scores_with_average.Score > golf_scores_with_average.AverageScore
    # Above or below average column changed to when above average = 1 and below average = 0
    golf_scores_with_average['AboveOrBelowAverage'] = golf_scores_with_average['AboveOrBelowAverage'].astype(int)
    golf_scores_with_average_num = golf_scores_with_average.drop(columns = ['DateOfRound'])
    base_features2 = ['Putts', 'PuttsInsideTenFeetAttempted', 'PuttsInsideTenFeetMade', 'PuttsInsideTenFeetMakePercentage', 'Fairways', 'Greens', 'Inside100', 'UpAndDownAttempts', 'UpAndDownSuccess', 'UpAndDownSuccessPercentage']
    X = golf_scores_with_average_num[base_features2]
    y = golf_scores_with_average_num.AboveOrBelowAverage
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1)
    model.fit(train_X, train_y)
    perm = PermutationImportance(model, random_state = 1).fit(val_X, val_y)
    importances = perm.feature_importances_
    importances = pd.DataFrame(importances)
    importances = importances.rename(index = {0 : 'Putts', 1: 'PuttsInsideTenFeetAttempted', 2: 'PuttsInsideTenFeetMade', 3:'PuttsInsideTenFeetMakePercentage' ,4: 'Fairways' , 5: 'Greens' , 6: 'Inside100'  ,7: 'UpAndDownAttempts' ,8: 'UpAndDownSuccess' , 9: 'UpAndDownSuccessPercentage' })
    #importances = importances.sort_values(by = [0], ascending = False)
    X_averages = X.mean()
    importances = importances.rename(columns = {0: 'Weights'})
    averages_with_weights = [importances, X_averages]
    averages_with_weights = pd.concat(averages_with_weights, axis = 1)
    averages_with_weights = averages_with_weights.rename(columns = {0 : 'Averages'})
    averages_with_weights = averages_with_weights.sort_values(by = 'Weights', ascending = False)
    round_data = round_x.reshape(-1,1)
    round_data = pd.DataFrame(round_data)
    round_data = round_data.rename(index = {0 : 'Putts', 1: 'PuttsInsideTenFeetAttempted', 2: 'PuttsInsideTenFeetMade', 3:'PuttsInsideTenFeetMakePercentage' ,4: 'Fairways' , 5: 'Greens' , 6: 'Inside100'  ,7: 'UpAndDownAttempts' ,8: 'UpAndDownSuccess' , 9: 'UpAndDownSuccessPercentage' })
    averages_with_weights_with_round = [averages_with_weights, round_data]
    averages_with_weights_with_round = pd.concat(averages_with_weights_with_round, axis = 1)
    averages_with_weights_with_round = averages_with_weights_with_round.rename(columns = {0 : 'Your Round'})
    averages_with_weights_with_round['Round Difference'] = averages_with_weights_with_round['Your Round'] - averages_with_weights_with_round['Averages']
    return averages_with_weights_with_round
     
  
def drill_suggestor(round_x):
    your_game = importance(round_x)
    if your_game.iloc[0,3] < 0 and your_game.iloc[1,3] < 0 and your_game.iloc[7,3] < 0 and your_game.iloc[8,3] < 0 and your_game.iloc[4,3] < 0 and your_game.iloc[5,3] > 0 and your_game.iloc[6,3] > 0 and your_game.iloc[9,3] > 0: 
        print("Work on everything")
    elif your_game.iloc[0,3] > 0 and your_game.iloc[1,3] > 0  and your_game.iloc[7,3] > 0 and your_game.iloc[8,3] > 0 and your_game.iloc[4,3] > 0 and your_game.iloc[5,3] < 0 and your_game.iloc[6,3] < 0 and your_game.iloc[9,3] < 0:
        print("Great day!")
    else:
        if your_game.iloc[0, 3] < 0:
            print("Work on chipping")
        elif your_game.iloc[1, 3] < 0:
            print("Work on irons")
        elif your_game.iloc[4,3] < 0:
            print("Hit the ball closer into the green")
        elif your_game.iloc[5,3] > 0:
            print("Hit more greens")
        elif your_game.iloc[6, 3] > 0:
            print("Make more putts")
        elif your_game.iloc[7, 3] < 0:
            print("Make more putts inside ten feet")
        elif your_game.iloc[8, 3] < 0:
            print("Hit driver straighter")
        elif your_game.iloc[9, 3] > 0:
            print("Work on wedges")
