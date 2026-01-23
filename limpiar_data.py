import pandas as pd
import numpy as np

df = pd.read_csv("diabetes_dataset.csv")

df_limpia = df.drop_duplicates().dropna().drop(columns=["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","diet_score","diabetes_risk_score","diabetes_stage","screen_time_hours_per_day",
"cardiovascular_history","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","sleep_hours_per_day","hypertension_history"]).reset_index(drop=True)

df_limpia.to_csv("diabetes_dataset_limpio.csv", index=False)