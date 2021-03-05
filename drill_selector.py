import pickle
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from drill_selection import putts_inside_of_ten, up_and_down, round_entry, predict, importance, drill_suggestor

mock_round = round_entry(32, 15, 10, 4, 4, 48, 14, 2)
print(predict(mock_round))
print(importance(mock_round))
drill_suggestor(mock_round)