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

mock_round = round_entry(35, 19, 17, 5, 11, 42, 7, 5)
drill_suggestor(mock_round)