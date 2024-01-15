import random
import pandas as pd
import numpy as np
import io
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

with open('20featureswithaccuracy(0.93).pkl', 'rb') as file:
    model = pickle.load(file)
    selected_symptoms = ['itching', 'skin_rash', 'joint_pain', 'stomach_pain', 'vomiting', 'burning_micturition',
                     'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'diarrhoea',
                     'mild_fever', 'phlegm', 'chest_pain', 'excessive_hunger', 'stiff_neck', 'loss_of_balance', 'muscle_pain',
                     'abnormal_menstruation']

user_symptoms = {symptom: random.choice(['yes', 'no']) for symptom in selected_symptoms}
user_symptoms = {symptom: 1 if response == 'yes' else 0 for symptom, response in user_symptoms.items()}
user_input_df = pd.DataFrame([user_symptoms])
prediction = model.predict(user_input_df)

# Print the result
print("Predicted Disease:")
print(prediction)                   
