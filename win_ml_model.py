from win_predictor import offensive_efficiency, efficient_points_scored, raw_EOP, game, predict
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

game = game(55, 17, 6, 1, 3, 44)

prediction = predict(game)