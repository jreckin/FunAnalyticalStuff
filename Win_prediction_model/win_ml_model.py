from win_predictor import offensive_efficiency, efficient_points_scored, raw_EOP, game, predict
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

game = game(50, 25, 11, 21, 12, 62)

prediction = predict(game)