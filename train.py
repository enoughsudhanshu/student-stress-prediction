#Step 1 Import all Lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#preprocessing 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#evaluation
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#Step 3 Data Load

df=pd.read_csv('student_stress_dataset_raw.csv')

#print(df.head())
#shape
#print("shape:-",df.shape)
#shape
#print(df.info())
#columns
#print("col:-",df.columns)
#SUM
#print(df.isnull().sum())

#Step 3 Split into 2 parts num and cat cols 

num_cols=[
    'Age',
    'Daily_Walk_km',
    'Screen_Time_hrs',
    'Sleep_hrs',
    'Study_Work_hrs',
]

cat_cols=[
    'Gender',
    'Activity_Level',
    'Diet_Quality',
    'Caffeine_Intake',
    
]


target='Stress_Level'

#step 4 Handle Missing Values (cat and num cols )


#filling missing value in num_cols
for col in num_cols:
    df[col].fillna(df[col].mean(),inplace=True)

#filling missing value in cat_cols
for col in cat_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)



#Step 5 converting all cat_cols in lower and proper format
df['Gender']=df['Gender'].str.lower().replace ({
    'm':'male',
    'f':'female'
    })

df['Activity_Level']=df['Activity_Level'].str.lower()
df['Diet_Quality']=df['Diet_Quality'].str.lower()
df['Caffeine_Intake']=df['Caffeine_Intake'].str.lower()
df['Stress_Level']=df['Stress_Level'].str.lower()

#print(df.info())

#Step 6 Encoding 
le=LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Encode target
df[target] = le.fit_transform(df[target])

print(df.info())
#Step 7
x=df.drop('Stress_Level',axis=1)
y = df['Stress_Level']

#Step 8

scaler =StandardScaler()

x[num_cols]=scaler.fit_transform(x[num_cols])

#Step 9

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("Train Shape:",x_train.shape)
print("Test Shape",x_test.shape)
print(x_test)


lr_model=LogisticRegression()
rf_model=RandomForestClassifier(random_state=42)

lr_model.fit(x_train,y_train)
rf_model.fit(x_train,y_train)

y_pred_lr = lr_model.predict(x_test)
y_pred_rf = rf_model.predict(x_test)


print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")

print("\nClassification Report: ")
print(classification_report(y_test,y_pred_lr))


print("\nConfusio Matrix: ")
print (confusion_matrix(y_test,y_pred_lr))

joblib.dump(lr_model, "stress_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")