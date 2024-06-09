#importing libraries
from shiny import App, render, ui,reactive
import shinyswatch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report
)
import pickle

# loading the model
# logit_model = pd.read_pickle('../models/logit_model_full.pkl')

# UI section starts from here 
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Heart Disease Prediction Model
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
                        ui.input_select("state","Which state do you live in?",
                                        {0: 'Alabama', 1: 'Alaska', 
                                         2: 'Arizona', 3: 'Arkansas',
                                         4: 'California', 5: 'Colorado',
                                         6: 'Connecticut', 7: 'Delaware',
                                         8: 'District of Columbia', 9: 'Florida',
                                         10: 'Georgia', 11: 'Guam', 12: 'Hawaii',
                                         13: 'Idaho', 14: 'Illinois',15: 'Indiana',
                                         16: 'Iowa', 17: 'Kansas', 18: 'Kentucky',
                                         19: 'Louisiana', 20: 'Maine',21: 'Maryland',
                                         22: 'Massachusetts', 23: 'Michigan',
                                         24: 'Minnesota', 25: 'Mississippi',
                                         26: 'Missouri', 27: 'Montana',
                                         28: 'Nebraska', 29: 'Nevada',
                                         30: 'New Hampshire',31: 'New Jersey',
                                         32: 'New Mexico', 33: 'New York',
                                         34: 'North Carolina',35: 'North Dakota',
                                         36: 'Ohio', 37: 'Oklahoma', 38: 'Oregon',
                                         39: 'Pennsylvania',40: 'Puerto Rico',
                                         41: 'Rhode Island', 42: 'South Carolina',
                                         43: 'South Dakota', 44: 'Tennessee',
                                         45: 'Texas', 46: 'Utah', 47: 'Vermont',
                                         48: 'Virgin Islands', 49: 'Virginia',
                                         50: 'Washington', 51: 'West Virginia',
                                         52: 'Wisconsin', 53: 'Wyoming'}),
                        ui.input_select("Sex", 
                                        "sex", 
                                        {0:"Female",
                                         1:"Male"}),
                        ui.input_select("age_category",
                                        "Age Group", 
                                          {0:'Age 18 to 24', 1:'Age 25 to 29', 
                                           2:'Age 30 to 34', 3:'Age 35 to 39',
                                           4:'Age 40 to 44', 5:'Age 45 to 49', 
                                           6:'Age 50 to 54', 7:'Age 55 to 59', 
                                           8:'Age 60 to 64', 9:'Age 65 to 69', 
                                           10:'Age 70 to 74', 11:'Age 75 to 79', 
                                           12:'Age 80 or older'}),
                        ui.input_numeric("weight", "Weight (in kilograms)",value = 0),
                        ui.input_numeric("height", "Height (in meters)",value = 0),
                        ui.input_select("smoker_status", "Smoking Status",
                                        {0: "Never smoked", 
                                         1: "Former smoker", 
                                         2: "Current smoker - now smokes some days",
                                         3: "Current smoker - now smokes every day"}),
                        ui.input_select("removedteeth", "Have you had any teeth removed?",
                                        {0: "No",1: "Yes"}),
                        ui.input_select("lastcheckup", "When was your checkup?",
                                        {0: "Within past year (anytime less than 12 months ago)",
                                         1: "Within past 2 years (1 year but less than 2 years ago)",
                                         2: "Within past 5 years (2 years but less than 5 years ago)",
                                         3: "5 or more years ago"}),
                        ui.input_select("general_health", "General Health", 
                                        {0: "Poor", 1: "Fair", 2: "Good",
                                         3: "Very good", 4: "Excellent"}),
                        ui.input_select("chest_scan","Have you ever had a chest scan?",
                                        {0: "No",1: "Yes"}),
                        ui.input_select("had_diabetes", "Do you have diabetes?",
                                        {0: "No",1: "Yes"}),
                        ui.input_select("difficulty_walking", "Difficulty walking",
                                        {0: "No", 1: "Yes"})
                         ),
        ui.panel_main(
            ui.markdown(
        """
        ### Heart Disease Risk Score (0 - 100)
        """
          ),
          #ui.output_text_verbatim("txt", placeholder=True),
          # a line of text to display the risk score
            ui.output_text_verbatim("txt", placeholder = "Risk Score"),
          # plot a distribution plot of the risk score
            #ui.output_plot("risk_score_plot"),
          # insert an image
            # ui.output_image("risk_score_plot", width = "60%", height = "30%")              
          )
    ),
)

## server section 

def server(input, output, session):
    @render.text("txt")
    def txt():
        return "Your risk score for CVD is 64.3"
    # @render.image("risk_score_plot")
    # def risk_score_plot():
    #     from pathlib import Path

    #     dir = Path(__file__).resolve().parent
    #     img: ImgData = {"src": str(dir / "output.png"), 
    #                     "width": "100%", "height": "100%"}
    #     return img

        

app = App(app_ui, server)