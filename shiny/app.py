from shiny import App, render, ui, reactive
from shinyswatch import theme
import numpy as np
import pandas as pd
import seaborn as sns
import requests
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

# read in pickle file from github
url_1 = 'https://raw.githubusercontent.com/hinyzee/heart-shiny/main/models/best_model.pkl'
response = requests.get(url_1)
best_model = pickle.loads(response.content)

url_2 = 'https://raw.githubusercontent.com/hinyzee/heart-shiny/main/models/preprocessor.pkl'
response = requests.get(url_2)
preprocessor = pickle.loads(response.content)

url_3 = 'https://raw.githubusercontent.com/hinyzee/heart-shiny/main/models/scaler.pkl'
response = requests.get(url_3)
standard_scaler = pickle.loads(response.content)



# Load the model
# model_path = "../models/best_model.pkl"
# with open(model_path, 'rb') as file:
#     best_model = pickle.load(file)

# with open('../models/preprocessor.pkl', 'rb') as file:
#     preprocessor = pickle.load(file)

# with open('../models/scaler.pkl', 'rb') as file:
#     standard_scaler = pickle.load(file)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', standard_scaler),
    ('model', best_model)
])

# Create a dictionary of unique values for selection inputs
unique_values = {
    'State': ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
              'Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Guam','Hawaii','Idaho', 'Illinois','Indiana','Iowa', 'Kansas', 'Kentucky',
               'Louisiana','Maine','Maryland','Massachusetts', 'Michigan',
               'Minnesota', 'Mississippi', 'Missouri', 'Montana','Nebraska', 
               'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
                'North Carolina','North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
                'Pennsylvania','Puerto Rico','Rhode Island', 'South Carolina',
                'South Dakota', 'Tennessee','Texas', 'Utah', 'Vermont',
                'Virgin Islands','Virginia','Washington','West Virginia',
                'Wisconsin','Wyoming'],
    'Sex': ['Female', 'Male'],
    'AgeCategory': [ 'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
                    'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
                    'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
                    'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
                    'Age 80 or older'],
    'LastCheckupTime': ["Within past year (anytime less than 12 months ago)", 
                        "Within past 2 years (1 year but less than 2 years ago)", 
                        "Within past 5 years (2 years but less than 5 years ago)", 
                        "5 or more years ago"],
    'PhysicalActivities': ['No', 'Yes'],
    'RemovedTeeth': ['None of them', '1 to 5', '6 or more, but not all', 'All'],
    'GeneralHealth': ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'],
    'SmokerStatus': ["Never smoked", "Former smoker", 
                     "Current smoker - now smokes some days", 
                     "Current smoker - now smokes every day"],
    'HIVTesting': ['No', 'Yes'],
    'FluVaxLast12': ['No', 'Yes'],
    'PneumoVaxEver': ['No', 'Yes'],
    'TetanusLast10Tdap': [ "No, did not receive any tetanus shot in the past 10 years", 
                          "Yes, received tetanus shot but not sure what type", 
                          "Yes, received Tdap in the past 10 years"],
    'HighRiskLastYear': ['No', 'Yes'],
    'CovidPos': ['No', 'Yes', 
                 'Tested positive using home test without a health professional'],
    'HadAsthma': ['No', 'Yes'],
    'HadSkinCancer': ['No', 'Yes'],
    'HadCOPD': ['No', 'Yes'],
    'HadDepressiveDisorder': ['No', 'Yes'],
    'HadKidneyDisease': ['No', 'Yes'],
    'HadArthritis': ['No', 'Yes'],
    'HadDiabetes': ['Yes', 'No', 
                    'No, pre-diabetes or borderline diabetes', 
                    'Yes, but only during pregnancy (female)'],
    'DeafOrHardOfHearing': ['No', 'Yes'],
    'BlindOrVisionDifficulty': ['No', 'Yes'],
    'DifficultyConcentrating': ['No', 'Yes'],
    'DifficultyWalking': ['No', 'Yes'],
    'DifficultyDressingBathing': ['No', 'Yes'],
    'DifficultyErrands': ['No', 'Yes'],
    'ChestScan': ['No', 'Yes'],
    'RaceEthnicityCategory': ['White only, Non-Hispanic', 
                              'Black only, Non-Hispanic',
                              'Other race only, Non-Hispanic', 
                              'Multiracial, Non-Hispanic',
                              'Hispanic'],
    'AlcoholDrinkers': ['No', 'Yes'],
    'ECigaretteUsage': ['Never used e-cigarettes in my entire life',
                            'Not at all (right now)',
                            'Use them some days',
                            'Use them every day']
}

# UI section starts from here 
app_ui = ui.page_fluid(
    theme.flatly(),
    ui.markdown(
        """
        ## Heart Disease Prediction Model
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("State", "Which state do you live in?", 
                            {i: state for i, state in enumerate(unique_values['State'])}),
            ui.input_select("Sex", "Sex", {i: sex for i, sex in enumerate(unique_values['Sex'])}),
            ui.input_select("AgeCategory", "Age Group", 
                            {i: age for i, age in enumerate(unique_values['AgeCategory'])}),
            ui.input_numeric("PhysicalHealthDays", "Physical Health Days", value=0),
            ui.input_numeric("MentalHealthDays", "Mental Health Days", value=0),
            ui.input_select("LastCheckupTime", "When was your checkup?", 
                            {i: checkup for i, checkup in enumerate(unique_values['LastCheckupTime'])}),
            ui.input_select("PhysicalActivities", "Physical Activities", 
                            {i: activity for i, activity in enumerate(unique_values['PhysicalActivities'])}),
            ui.input_numeric("SleepHours", "Sleep Hours", value=0),
            ui.input_select("RemovedTeeth", "Have you had any teeth removed?", 
                            {i: teeth for i, teeth in enumerate(unique_values['RemovedTeeth'])}),
            ui.input_select("GeneralHealth", "General Health", 
                            {i: health for i, health in enumerate(unique_values['GeneralHealth'])}),
            ui.input_numeric("HeightInMeters", "Height (in meters)", value=0),
            ui.input_numeric("WeightInKilograms", "Weight (in kilograms)", value=0),
            ui.input_select("SmokerStatus", "Smoking Status", 
                            {i: smoker for i, smoker in enumerate(unique_values['SmokerStatus'])}),
            ui.input_select("HIVTesting", "HIV Testing", 
                            {i: hiv for i, hiv in enumerate(unique_values['HIVTesting'])}),
            ui.input_select("FluVaxLast12", "Flu Vax Last 12 Months", 
                            {i: flu for i, flu in enumerate(unique_values['FluVaxLast12'])}),
            ui.input_select("PneumoVaxEver", "Pneumo Vax Ever", 
                            {i: pneumo for i, pneumo in enumerate(unique_values['PneumoVaxEver'])}),
            ui.input_select("TetanusLast10Tdap", "Tetanus Last 10 Years Tdap", 
                            {i: tetanus for i, tetanus in enumerate(unique_values['TetanusLast10Tdap'])}),
            ui.input_select("HighRiskLastYear", "High Risk Last Year", 
                            {i: risk for i, risk in enumerate(unique_values['HighRiskLastYear'])}),
            ui.input_select("CovidPos", "Covid Positive", 
                            {i: covid for i, covid in enumerate(unique_values['CovidPos'])}),
            ui.input_select("HadAsthma", "Had Asthma", 
                            {i: asthma for i, asthma in enumerate(unique_values['HadAsthma'])}),
            ui.input_select("HadSkinCancer", "Had Skin Cancer", 
                            {i: cancer for i, cancer in enumerate(unique_values['HadSkinCancer'])}),
            ui.input_select("HadCOPD", "Had COPD", 
                            {i: copd for i, copd in enumerate(unique_values['HadCOPD'])}),
            ui.input_select("HadDepressiveDisorder", "Had Depressive Disorder", 
                            {i: depressive for i, depressive in enumerate(unique_values['HadDepressiveDisorder'])}),
            ui.input_select("HadKidneyDisease", "Had Kidney Disease", 
                            {i: kidney for i, kidney in enumerate(unique_values['HadKidneyDisease'])}),
            ui.input_select("HadArthritis", "Had Arthritis", 
                            {i: arthritis for i, arthritis in enumerate(unique_values['HadArthritis'])}),
            ui.input_select("HadDiabetes", "Had Diabetes", 
                            {i: diabetes for i, diabetes in enumerate(unique_values['HadDiabetes'])}),
            ui.input_select("DeafOrHardOfHearing", "Deaf or Hard of Hearing", 
                            {i: deaf for i, deaf in enumerate(unique_values['DeafOrHardOfHearing'])}),
            ui.input_select("BlindOrVisionDifficulty", "Blind or Vision Difficulty", 
                            {i: blind for i, blind in enumerate(unique_values['BlindOrVisionDifficulty'])}),
            ui.input_select("DifficultyConcentrating", "Difficulty Concentrating", 
                            {i: concentrate for i, concentrate in enumerate(unique_values['DifficultyConcentrating'])}),
            ui.input_select("DifficultyWalking", "Difficulty Walking", 
                            {i: walk for i, walk in enumerate(unique_values['DifficultyWalking'])}),
            ui.input_select("DifficultyDressingBathing", "Difficulty Dressing or Bathing", 
                            {i: dress for i, dress in enumerate(unique_values['DifficultyDressingBathing'])}),
            ui.input_select("DifficultyErrands", "Difficulty Running Errands", 
                            {i: errand for i, errand in enumerate(unique_values['DifficultyErrands'])}),
            ui.input_select("ChestScan", "Have you ever had a chest scan?", 
                            {i: scan for i, scan in enumerate(unique_values['ChestScan'])}),
            ui.input_select("RaceEthnicityCategory", "Race/Ethnicity Category", 
                            {i: race for i, race in enumerate(unique_values['RaceEthnicityCategory'])}),
            ui.input_select("AlcoholDrinkers", "Alcohol Drinkers", 
                            {i: alcohol for i, alcohol in enumerate(unique_values['AlcoholDrinkers'])}),
            ui.input_select("ECigaretteUsage", "E-Cigarette Usage",
                            {i: ecig for i, ecig in enumerate(unique_values['ECigaretteUsage'])}),
            ui.input_action_button("btn", "Predict"),

        ),
        ui.panel_main(
            ui.markdown(
                """
                ### Heart Disease Risk Score (0 - 100)
                """
            ),
            ui.output_text_verbatim("txt", placeholder="Risk Score"),
            ui.output_plot("risk_score_plot", height="30%", width="100%")

        )
    )
)

def server(input, output, session):
    @output
    @render.text
    @reactive.event(input.btn)
    def txt():
        # Collect input data
        input_data = pd.DataFrame([{
            'State': unique_values['State'][unique_values['State'] == input.State()],
            'Sex': unique_values['Sex'][unique_values['Sex'] == input.Sex()],
            'AgeCategory': unique_values['AgeCategory'][unique_values['AgeCategory'] == input.AgeCategory()],
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'MentalHealthDays': input.MentalHealthDays(),
            'LastCheckupTime': unique_values['LastCheckupTime'][unique_values['LastCheckupTime'] == input.LastCheckupTime()],
            'PhysicalActivities': unique_values['PhysicalActivities'][unique_values['PhysicalActivities'] == input.PhysicalActivities()],
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': unique_values['RemovedTeeth'][unique_values['RemovedTeeth'] == input.RemovedTeeth()],
            'GeneralHealth': unique_values['GeneralHealth'][unique_values['GeneralHealth'] == input.GeneralHealth()],
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'SmokerStatus': unique_values['SmokerStatus'][unique_values['SmokerStatus'] == input.SmokerStatus()],
            'HIVTesting': unique_values['HIVTesting'][unique_values['HIVTesting'] == input.HIVTesting()],
            'FluVaxLast12': unique_values['FluVaxLast12'][unique_values['FluVaxLast12'] == input.FluVaxLast12()],
            'PneumoVaxEver': unique_values['PneumoVaxEver'][unique_values['PneumoVaxEver'] == input.PneumoVaxEver()],
            'TetanusLast10Tdap': unique_values['TetanusLast10Tdap'][unique_values['TetanusLast10Tdap'] == input.TetanusLast10Tdap()],
            'HighRiskLastYear': unique_values['HighRiskLastYear'][unique_values['HighRiskLastYear'] == input.HighRiskLastYear()],
            'CovidPos': unique_values['CovidPos'][unique_values['CovidPos'] == input.CovidPos()],
            'HadAsthma': unique_values['HadAsthma'][unique_values['HadAsthma'] == input.HadAsthma()],
            'HadSkinCancer': unique_values['HadSkinCancer'][unique_values['HadSkinCancer'] == input.HadSkinCancer()],   
            'HadCOPD': unique_values['HadCOPD'][unique_values['HadCOPD'] == input.HadCOPD()],
            'HadDepressiveDisorder': unique_values['HadDepressiveDisorder'][unique_values['HadDepressiveDisorder'] == input.HadDepressiveDisorder()],   
            'HadKidneyDisease': unique_values['HadKidneyDisease'][unique_values['HadKidneyDisease'] == input.HadKidneyDisease()],
            'HadArthritis': unique_values['HadArthritis'][unique_values['HadArthritis'] == input.HadArthritis()],
            'HadDiabetes': unique_values['HadDiabetes'][unique_values['HadDiabetes'] == input.HadDiabetes()],
            'DeafOrHardOfHearing': unique_values['DeafOrHardOfHearing'][unique_values['DeafOrHardOfHearing'] == input.DeafOrHardOfHearing()],   
            'BlindOrVisionDifficulty': unique_values['BlindOrVisionDifficulty'][unique_values['BlindOrVisionDifficulty'] == input.BlindOrVisionDifficulty()],   
            'DifficultyConcentrating': unique_values['DifficultyConcentrating'][unique_values['DifficultyConcentrating'] == input.DifficultyConcentrating()],   
            'DifficultyWalking': unique_values['DifficultyWalking'][unique_values['DifficultyWalking'] == input.DifficultyWalking()],   
            'DifficultyDressingBathing': unique_values['DifficultyDressingBathing'][unique_values['DifficultyDressingBathing'] == input.DifficultyDressingBathing()],   
            'DifficultyErrands': unique_values['DifficultyErrands'][unique_values['DifficultyErrands'] == input.DifficultyErrands()],   
            'ChestScan': unique_values['ChestScan'][unique_values['ChestScan'] == input.ChestScan()],
            'RaceEthnicityCategory': unique_values['RaceEthnicityCategory'][unique_values['RaceEthnicityCategory'] == input.RaceEthnicityCategory()],
            'ECigaretteUsage': unique_values['ECigaretteUsage'][unique_values['ECigaretteUsage'] == input.ECigaretteUsage()],
            'AlcoholDrinkers': unique_values['AlcoholDrinkers'][unique_values['AlcoholDrinkers'] == input.AlcoholDrinkers()],
            'BMI': input.WeightInKilograms() / (input.HeightInMeters() ** 2)
        }])

        prediction = pipeline.predict_proba(input_data)[0][1] * 100
        return f"Your risk score for CVD is {prediction:.2f}"
    
    # Plot the distribution of the probability of having a heart disease
    # Generate a random sample of 1000 predictions
    @output
    @render.plot(height=500, width=600)
    @reactive.event(input.btn)
    def risk_score_plot():
        # make the prediction
        input_data = pd.DataFrame([{
            'State': unique_values['State'][unique_values['State'] == input.State()],
            'Sex': unique_values['Sex'][unique_values['Sex'] == input.Sex()],
            'AgeCategory': unique_values['AgeCategory'][unique_values['AgeCategory'] == input.AgeCategory()],
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'MentalHealthDays': input.MentalHealthDays(),
            'LastCheckupTime': unique_values['LastCheckupTime'][unique_values['LastCheckupTime'] == input.LastCheckupTime()],
            'PhysicalActivities': unique_values['PhysicalActivities'][unique_values['PhysicalActivities'] == input.PhysicalActivities()],
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': unique_values['RemovedTeeth'][unique_values['RemovedTeeth'] == input.RemovedTeeth()],
            'GeneralHealth': unique_values['GeneralHealth'][unique_values['GeneralHealth'] == input.GeneralHealth()],
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'SmokerStatus': unique_values['SmokerStatus'][unique_values['SmokerStatus'] == input.SmokerStatus()],
            'HIVTesting': unique_values['HIVTesting'][unique_values['HIVTesting'] == input.HIVTesting()],
            'FluVaxLast12': unique_values['FluVaxLast12'][unique_values['FluVaxLast12'] == input.FluVaxLast12()],
            'PneumoVaxEver': unique_values['PneumoVaxEver'][unique_values['PneumoVaxEver'] == input.PneumoVaxEver()],
            'TetanusLast10Tdap': unique_values['TetanusLast10Tdap'][unique_values['TetanusLast10Tdap'] == input.TetanusLast10Tdap()],
            'HighRiskLastYear': unique_values['HighRiskLastYear'][unique_values['HighRiskLastYear'] == input.HighRiskLastYear()],
            'CovidPos': unique_values['CovidPos'][unique_values['CovidPos'] == input.CovidPos()],
            'HadAsthma': unique_values['HadAsthma'][unique_values['HadAsthma'] == input.HadAsthma()],
            'HadSkinCancer': unique_values['HadSkinCancer'][unique_values['HadSkinCancer'] == input.HadSkinCancer()],   
            'HadCOPD': unique_values['HadCOPD'][unique_values['HadCOPD'] == input.HadCOPD()],
            'HadDepressiveDisorder': unique_values['HadDepressiveDisorder'][unique_values['HadDepressiveDisorder'] == input.HadDepressiveDisorder()],   
            'HadKidneyDisease': unique_values['HadKidneyDisease'][unique_values['HadKidneyDisease'] == input.HadKidneyDisease()],
            'HadArthritis': unique_values['HadArthritis'][unique_values['HadArthritis'] == input.HadArthritis()],
            'HadDiabetes': unique_values['HadDiabetes'][unique_values['HadDiabetes'] == input.HadDiabetes()],
            'DeafOrHardOfHearing': unique_values['DeafOrHardOfHearing'][unique_values['DeafOrHardOfHearing'] == input.DeafOrHardOfHearing()],   
            'BlindOrVisionDifficulty': unique_values['BlindOrVisionDifficulty'][unique_values['BlindOrVisionDifficulty'] == input.BlindOrVisionDifficulty()],   
            'DifficultyConcentrating': unique_values['DifficultyConcentrating'][unique_values['DifficultyConcentrating'] == input.DifficultyConcentrating()],   
            'DifficultyWalking': unique_values['DifficultyWalking'][unique_values['DifficultyWalking'] == input.DifficultyWalking()],   
            'DifficultyDressingBathing': unique_values['DifficultyDressingBathing'][unique_values['DifficultyDressingBathing'] == input.DifficultyDressingBathing()],   
            'DifficultyErrands': unique_values['DifficultyErrands'][unique_values['DifficultyErrands'] == input.DifficultyErrands()],   
            'ChestScan': unique_values['ChestScan'][unique_values['ChestScan'] == input.ChestScan()],
            'RaceEthnicityCategory': unique_values['RaceEthnicityCategory'][unique_values['RaceEthnicityCategory'] == input.RaceEthnicityCategory()],
            'ECigaretteUsage': unique_values['ECigaretteUsage'][unique_values['ECigaretteUsage'] == input.ECigaretteUsage()],
            'AlcoholDrinkers': unique_values['AlcoholDrinkers'][unique_values['AlcoholDrinkers'] == input.AlcoholDrinkers()],
            'BMI': input.WeightInKilograms() / (input.HeightInMeters() ** 2)
        }])
        prediction = pipeline.predict_proba(input_data)[0][1] * 100
        # similate 1000 predictions that skew to the left with min 0 and max 100
        risk_scores = np.random.beta(2, 5, 1000) * 100
        plt.figure(figsize=(5, 5))
        sns.histplot(risk_scores, kde=True, color = "gray")
        # Add a vertical line to indicate the user's risk score
        plt.axvline(prediction, color='red', linestyle='--', label='Your Risk Score')
        plt.xlabel('Risk Score')
        plt.ylabel('Probability Density')
        # hide y ticks
        plt.yticks([])
        plt.suptitle('Distribution of Heart Disease Risk Scores among the population')
        plt.title('The red dashed line indicates your risk score for heart disease')
        return plt.gcf()                                
    
app = App(app_ui, server)
